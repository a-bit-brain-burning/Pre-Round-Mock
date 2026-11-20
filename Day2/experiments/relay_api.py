"""Authenticated competition client; cookies and request snapshots stay in .cache.

Examples:
  python3 experiments/relay_api.py get /api/me
  python3 experiments/relay_api.py submit argmax 2/code.cpp --note 'candidate'
  python3 experiments/relay_api.py wait TASK_ID
"""
import argparse
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "relay"
TERMINAL = {"done", "rejected", "cancelled", "resolved"}


def relay_base():
    config = CACHE / "base-url"
    base = os.environ.get("PREROUND_RELAY_URL")
    if not base and config.is_file():
        base = config.read_text(encoding="utf-8")
    if not base or not base.strip():
        raise RuntimeError("Set PREROUND_RELAY_URL or save your team relay URL in Day2/.cache/relay/base-url")
    base = base.strip().rstrip("/")
    url = urllib.parse.urlsplit(base)
    if url.scheme != "https" or not url.hostname or url.username or url.query or url.fragment:
        raise RuntimeError("The relay URL must be an HTTPS base URL without credentials, queries or fragments")
    return base


def request(path, payload=None):
    base = relay_base()
    cookies = http.cookiejar.MozillaCookieJar(str(CACHE / "cookies.txt"))
    cookies.load(ignore_discard=True, ignore_expires=True)
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookies))
    headers = {"X-Relay-Request": "1", "Content-Type": "application/json"}
    data = None if payload is None else json.dumps(payload).encode()
    # Retries reuse the exact payload, including client_key, for idempotency.
    for attempt in range(3):
        try:
            req = urllib.request.Request(base + path, data=data, headers=headers)
            with opener.open(req, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2)


def summary(job):
    result = job.get("result")
    if isinstance(result, str):
        result = json.loads(result)
    result = result or {}
    score = result.get("contest_score_units")
    print(json.dumps({
        "id": job.get("id"), "state": job.get("state"),
        "status": result.get("status", job.get("status")),
        "notice": job.get("notice"), "sha256": job.get("sha256"),
        "total_us": None if result.get("total_runtime_ms") is None
                    else result["total_runtime_ms"] * 1000,
        "points": None if score is None else score / 10000,
        "cases_us": [round(test["runtime_ms"] * 1000, 3)
                     for test in result.get("test_results", [])],
    }, ensure_ascii=False), flush=True)
    if job.get("state") in TERMINAL and result.get("status") not in {"ACCEPTED", "PASSED"}:
        print(json.dumps(result, ensure_ascii=False), flush=True)


def wait(job_id):
    previous = None
    while True:
        job = request(f"/api/jobs/{job_id}")
        (CACHE / f"job-{job_id}.json").write_text(json.dumps(job, ensure_ascii=False, indent=2))
        state = job.get("state")
        signature = (state, job.get("status"))
        if signature != previous:
            summary(job)
            previous = signature
        if state == "blocked":
            raise RuntimeError(f"Queue blocked: {job.get('notice')}; stop optimization.")
        if state in TERMINAL:
            return job
        time.sleep(4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("get")
    get.add_argument("path")
    poll = sub.add_parser("wait")
    poll.add_argument("job_id")
    for name in ("run", "submit"):
        evaluate = sub.add_parser(name)
        evaluate.add_argument("problem", choices=["avg-pool-3d", "batch-norm", "box-blur"])
        evaluate.add_argument("source", type=Path)
        evaluate.add_argument("--language", choices=["cuda", "python", "cute"], default=None)
        evaluate.add_argument("--contest", default="practice-2026-11-17")
        evaluate.add_argument("--note", default="")
    args = parser.parse_args()
    if args.command == "get":
        print(json.dumps(request(args.path), ensure_ascii=False, indent=2))
    elif args.command == "wait":
        wait(args.job_id)
    else:
        me = request("/api/me")
        if me.get("counts", {}).get("blocked", 0):
            raise RuntimeError("Queue has blocked tasks; stop optimization.")
        code = args.source.read_text()
        client_key = str(uuid.uuid4())
        language = args.language or ("python" if args.source.suffix == ".py" else "cuda")
        payload = dict(contest=args.contest, problem=args.problem, language=language,
                       mode=args.command, code=code, note=args.note,
                       client_key=client_key, start_queue=True)
        (CACHE / f"request-{client_key}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(f"source={args.source} sha256={hashlib.sha256(code.encode()).hexdigest()}", flush=True)
        created = request("/api/jobs", payload)
        print(json.dumps(created), flush=True)
        wait(created["id"])


if __name__ == "__main__":
    main()
