import hashlib
import re
from collections import defaultdict
from functools import lru_cache
from math import erf, log, prod, sqrt
from statistics import median

import numpy as np

from contest_scoring import score_times
from history_model import fingerprint


@lru_cache(maxsize=100000)
def line_hash(tokens):
    return int.from_bytes(hashlib.blake2b(b" ".join(tokens),digest_size=8).digest(),"big")


def signature(code):
    lines, launch = set(), set()
    for line in code.splitlines():
        stripped=line.strip()
        if not stripped or stripped.startswith((b"//",b"#include")):
            continue
        tokens=tuple(re.findall(rb"\w+|[^\w\s]",stripped))
        if len(tokens)<3:
            continue
        value=line_hash(tokens)
        lines.add(value)
        if any(t in tokens for t in (b"if",b"case",b"constexpr",b"launch",b"return",b"num_warps")) or b"<<<" in line:
            launch.add(value)
    return fingerprint(code),frozenset(lines),frozenset(launch)


@lru_cache(maxsize=50000)
def similarity(left,right):
    values=[len(a&b)/len(a|b) if a or b else 1.0 for a,b in zip(left,right,strict=True)]
    return (2*values[0]+2*values[1]+values[2])/5


@lru_cache(maxsize=10000)
def edits(before, after):
    return (after[1]-before[1],before[1]-after[1],after[2]-before[2],before[2]-after[2])


def paired_change(nearest, features):
    anchor=nearest[0]
    change=edits(anchor[2],features)
    if not any(change):
        return None
    pairs=[]
    for left in nearest[:8]:
        for right in nearest[:8]:
            if left[1]==right[1]:
                continue
            previous=edits(left[2],right[2])
            shared=sum(len(a&b) for a,b in zip(change,previous,strict=True))
            total=sum(len(a|b) for a,b in zip(change,previous,strict=True))
            if total and shared>=2 and shared/total>=0.2:
                pairs.append((shared/total,left,right))
    if not pairs:
        return None
    pairs.sort(key=lambda p:p[0],reverse=True)
    closest=[p for p in pairs[:3] if p[0]>=pairs[0][0]-0.1]
    delta=np.median([np.log(p[2][3])-np.log(p[1][3]) for p in closest],axis=0)
    return pairs[0][0],np.log(anchor[3])+delta


class LocalTimingModel:
    def __init__(self):
        self.observations=defaultdict(dict)

    def observe(self, group, source, features, receipt, times, forecast):
        residual=None if forecast is None else [
            log(t/u) for t,u in zip(times,forecast["raw_us"],strict=True)]
        self.observations[group][source]=(features,receipt,times,residual)

    def predict(self, group, source, features, rule, cases):
        neighbors=[]
        for key,(other,rid,times,error) in self.observations[group].items():
            if key!=source:
                match=similarity(features,other)
                if match>=0.25:
                    neighbors.append((match,rid,other,times,error))
        if not neighbors:
            return None
        neighbors.sort(key=lambda n:(n[0],n[1]),reverse=True)
        nearest=neighbors[:12]
        if nearest[0][0]<0.4:
            return None
        times=np.log([n[3] for n in nearest])
        center=np.median(times[:3],axis=0)
        kernel=np.array([[similarity(a[2],b[2])**3 for b in nearest] for a in nearest])
        # 对同一份源码的小幅修改估计逐用例时间差，正则项限制单次测量噪声的影响。
        kernel.flat[::len(nearest)+1]+=0.06
        weights=np.linalg.solve(kernel,np.array([n[0]**3 for n in nearest]))
        extent=np.ptp(times[:3],axis=0)/4
        raw=np.clip(center+weights@(times-center),times.min(axis=0)-extent,times.max(axis=0)+extent)
        paired=paired_change(nearest,features)
        if paired:
            weight,paired_time=paired
            raw=(1-weight)*raw+weight*paired_time
        past=[n for n in nearest if n[4] is not None]
        predicted=raw.copy()
        if past:
            errors=np.array([n[4] for n in past])
            predicted+=len(past)/(len(past)+3)*np.median(errors,axis=0)
        estimate=np.exp(predicted).tolist()
        low=high=None
        if len(past)>=3:
            spread=np.max(np.abs(errors),axis=0)
            low=score_times(rule,cases,np.exp(predicted+spread).tolist())
            high=score_times(rule,cases,np.exp(predicted-spread).tolist())
        raw_us=np.exp(raw).tolist()
        draws=[score_times(rule,cases,(np.exp(raw+np.array(n[4]))).tolist()) for n in past]
        return {"score":score_times(rule,cases,estimate),"score_low":low,"score_high":high,"score_draws":draws,
                "raw_score":score_times(rule,cases,raw_us),"raw_us":raw_us,"estimate_us":estimate,
                "references":[n[1] for n in nearest],"similarities":[n[0] for n in nearest],
                "calibration_receipts":[n[1] for n in past]}


def gain_signal(prediction, incumbent, probe_margin=0):
    if incumbent>=1500:
        return {"gain_probability":0,"small_probability":0,"gain_signal":False,"small_signal":False}
    draws=prediction.get("score_draws",[])
    point=prediction["score"]
    if len(draws)<3:
        return {"gain_signal":point>incumbent,"small_signal":0<point-incumbent<=10}
    mid=median(draws)
    deviation=1.4826*median(abs(s-mid) for s in draws)
    scale=max(1.0,probe_margin,deviation/len(draws)**0.2)
    def cdf(threshold):
        return sum((1+erf((threshold-s)/(sqrt(2)*scale)))/2 for s in draws)/len(draws)
    below=cdf(incumbent)
    gain=1-below
    small=(1 if incumbent+10>=1500 else cdf(incumbent+10))-below
    return {"gain_probability":gain,"small_probability":small,"gain_signal":gain>=1/3,
            "small_signal":0<point-incumbent<=10 or (gain>=1/3 and small>=0.15)}


def apply_probe_spread(prediction, fractions, rule, cases):
    times=prediction["estimate_us"]
    lower=score_times(rule,cases,[t*(1+w) for t,w in zip(times,fractions,strict=True)])
    upper=score_times(rule,cases,[t/(1+w) for t,w in zip(times,fractions,strict=True)])
    # 探针给出同一配置的重复计时差，模型还需要历史预测误差才能给出完整区间。
    if prediction["score_high"] is not None:
        prediction["score_low"]=min(prediction["score_low"],lower)
        prediction["score_high"]=max(prediction["score_high"],upper)
    return max(upper-prediction["score"],prediction["score"]-lower)


def available_probe(details, problem, names, cutoff):
    if problem not in details:
        return None
    if problem=="elu":
        sizes=[prod(map(int,n.split("x"))) for n in names]
    else:
        sizes=[int(n.removeprefix("N = ").replace(",","")) for n in names]
    for ms,job,widths in reversed(details[problem]):
        if ms<cutoff and all(str(s) in widths for s in sizes):
            return job,[widths[str(s)] for s in sizes]
