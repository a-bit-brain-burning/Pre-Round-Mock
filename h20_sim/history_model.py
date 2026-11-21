import hashlib
import argparse
import gzip
import json
import re
from collections import defaultdict
from functools import lru_cache
from heapq import nsmallest
from statistics import median
from datetime import datetime
from pathlib import Path

from contest_scoring import score_times


@lru_cache(maxsize=100000)
def gram_hash(words):
    return int.from_bytes(hashlib.blake2b(b" ".join(words), digest_size=8).digest(), "big")


def fingerprint(code):
    # 这些连续符号只用于查找相似源码，实际计时来自已完成的同题 H20 回执。
    words = re.findall(rb"\w+|[^\w\s]", code)
    features = {gram_hash(tuple(words[i:i+5])) for i in range(len(words)-4)}
    return frozenset(nsmallest(128, features))


class TimingBlend:
    def __init__(self):
        self.errors = defaultdict(dict)

    def observe(self, group, source, receipt, actual, direct, corrected):
        if source not in self.errors[group]:
            self.errors[group][source] = (receipt, [abs(t-u) for t,u in zip(actual,direct,strict=True)],
                                         [abs(t-u) for t,u in zip(actual,corrected,strict=True)])

    def predict(self, group, source, direct, corrected):
        past = [v for key,v in self.errors[group].items() if key!=source]
        if len(past)<2:
            return corrected, []
        result = []
        for i,(a,b) in enumerate(zip(direct,corrected,strict=True)):
            first = median(v[1][i] for v in past)
            second = median(v[2][i] for v in past)
            result.append((second*a+first*b)/(first+second) if first+second else b)
        return result, [v[0] for v in past]


class HistoryModel:
    def __init__(self):
        self.observations = defaultdict(dict)
        self.residuals = defaultdict(dict)

    def observe(self, group, source, features, receipt, times, forecast):
        self.observations[group][source] = (features, receipt, times)
        if forecast and source not in self.residuals[group]:
            self.residuals[group][source] = (receipt, [actual / raw for actual, raw in
                zip(times, forecast["raw_us"], strict=True)], features)

    def predict(self, group, source, features, rule, cases):
        neighbors = []
        for key, (other, receipt, times) in self.observations[group].items():
            if key == source:
                continue
            common = len(other & features)
            similarity = common / max(1, len(other) + len(features) - common)
            if similarity >= 0.5:
                neighbors.append((similarity, receipt, times))
        if not neighbors:
            return None
        neighbors.sort(reverse=True)
        # 保留最接近的三份源码，相似度相差超过 0.1 的参照不参与同一次估计。
        selected = [n for n in neighbors[:3] if n[0] >= neighbors[0][0] - 0.1]
        raw = [median(v) for v in zip(*(n[2] for n in selected), strict=True)]
        past = []
        for key, record in self.residuals[group].items():
            if key != source:
                overlap = len(features & record[2])
                if 2 * overlap >= len(features) + len(record[2]) - overlap:
                    past.append(record)
        ratios = list(zip(*(v[1] for v in past), strict=True)) if past else []
        predicted = [t * median(rr) for t, rr in zip(raw, ratios)] if past else raw
        low = high = None
        if len(past) >= 2:
            # 逐份留出计算历史倍数误差。每份源码只占一个观测值。
            widths = []
            for rr in ratios:
                ordered = sorted(rr)
                below, above = median(ordered[1:]), median(ordered[:-1])
                widths.append(max(below / ordered[0], ordered[-1] / above, 1.0))
            low = score_times(rule, cases, [t * w for t, w in zip(predicted, widths, strict=True)])
            high = score_times(rule, cases, [t / w for t, w in zip(predicted, widths, strict=True)])
        return {"score": score_times(rule, cases, predicted), "score_low": low, "score_high": high,
                "raw_score": score_times(rule, cases, raw), "raw_us": raw, "estimate_us": predicted,
                "references": [n[1] for n in selected], "similarities": [n[0] for n in selected],
                "calibration_receipts": [v[0] for v in past]}


def main():
    parser = argparse.ArgumentParser(description="Estimate a new source from earlier completed H20 submissions")
    parser.add_argument("--history", required=True, type=Path)
    source_args=parser.add_mutually_exclusive_group(required=True)
    source_args.add_argument("--source",type=Path)
    source_args.add_argument("--source-receipt",type=Path)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--as-of", required=True, type=datetime.fromisoformat)
    args = parser.parse_args()
    if args.as_of.tzinfo is None:
        raise ValueError("An explicit timezone is required")
    cutoff = round(args.as_of.timestamp()*1000)
    with gzip.open(args.history, "rt", encoding="utf8") as stream:
        archive = json.load(stream)
    if archive["version"] not in (1,2):
        raise ValueError("Unsupported history format")
    history = HistoryModel()
    identify=fingerprint
    local=archive["version"]==2
    if local:
        from local_timing import LocalTimingModel, apply_probe_spread, available_probe, gain_signal, signature
        from submission_pacing import SubmissionPacer
        history,identify=LocalTimingModel(),signature
    if args.source_receipt is None:
        source=args.source.read_bytes()
    else:
        receipt=json.loads(args.source_receipt.read_text(encoding="utf-8-sig"))
        source=receipt["code"].encode("utf8")
        if hashlib.sha256(source).hexdigest()!=receipt["sha256"]:
            raise ValueError("Source snapshot SHA-256 mismatch")
    key = hashlib.sha256(source).hexdigest()
    seen = set()
    incumbent = max((r["score"] for r in archive["scores"] if r["problem"]==args.problem and
                     r["available_ms"]<cutoff),default=0)
    for row in archive["observations"]:
        if row["problem"]!=args.problem or row["available_ms"]>=cutoff or row["key"]==key:
            continue
        if row["id"] in seen:
            raise ValueError("Duplicate observation")
        seen.add(row["id"])
        if row["language"]!=args.language or row["protocol"]!=args.protocol:
            continue
        group = (args.problem,args.language,args.protocol,tuple(row["cases"]))
        pred = {"raw_us":row["raw_us"]} if row["raw_us"] is not None else None
        feature_data=archive["fingerprints"][row["key"]]
        features=tuple(map(frozenset,feature_data)) if local else frozenset(feature_data)
        history.observe(group,row["key"],features,row["id"],row["us"],pred)
    groups = list(history.observations)
    if len(groups)>1:
        raise ValueError("Case schedule changed within this timing protocol")
    out = None
    if groups:
        group, = groups
        rule = {"mode":"practice_total","base_points":1000,"benchmark_total_us":archive["benchmark_us"][args.problem],
                "case_ids":list(group[3])}
        cases=[{"id":c} for c in group[3]]
        out = history.predict(group,key,identify(source),rule,cases)
        if local and out:
            probe=available_probe(archive["probes"],args.problem,group[3],cutoff)
            if probe:
                job,spreads=probe
                out["jobs"]=[job]
                out["probe_margin"]=apply_probe_spread(out,spreads,rule,cases)
            out.update(gain_signal(out,incumbent,out.get("probe_margin",0)))
    out = out or {"score":None,"score_low":None,"score_high":None}
    out["incumbent"] = incumbent
    out["decision"] = "KEEP"
    if incumbent==1500:
        out["decision"]="SKIP_CAP"
    elif out["score_high"] is not None and out["score_high"]<=incumbent:
        out["decision"]="SKIP_MODEL"
    elif out["score_low"] is not None and out["score_low"]>incumbent:
        out["decision"]="PRIORITIZE"
    if local:
        contest=archive["contest_days"].get(args.problem)
        if contest is not None:
            pacer=SubmissionPacer()
            repeat=False
            for record in archive["pacing"]:
                if record["day"]!=contest or record["start_ms"]>=cutoff:
                    continue
                repeat|=record["problem"]==args.problem and record["sha256"]==key
                if record["repeat"]:
                    continue
                if pacer.start_ms is None:
                    pacer.start_ms=record["start_ms"]
                pacer.seen+=1
                if record["action"]=="KEEP":
                    pacer.used[record["problem"]]+=1
            out["used_before"]=pacer.used.total()
            out["budget_decision"]=pacer.decide(args.problem,repeat,incumbent,out["score"],out["decision"],
                                                cutoff,out.get("probe_margin",0),out.get("small_signal",False),
                                                out.get("gain_signal",False),out["score_high"])
    print(json.dumps(out,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
