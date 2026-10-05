"""Run StarGate against a live model over the API.

The original runner loads MiniMind weights directly, so it could only ever score
that one model. This talks to the chat endpoint, which means it can score
GTC-2.5 Turbo and can be re-run after every SFT to see whether a change helped.

    python3 run_stargate.py --cases cases_v2.jsonl --model tfmf --key <token>

Every check states its own reason on failure, so a bad score can be traced to the
case that produced it rather than being taken on trust.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_API = os.getenv("TAI_API_BASE", "http://127.0.0.1:5001")
DEFAULT_KEY = os.getenv("TAI_API_KEY") or os.getenv("TAI_INTERNAL_TOKEN") or ""


def ask(api, key, model, prompt, max_tokens, thinking=False, temperature=0.0, timeout=180):
    body = json.dumps({
        "text": prompt, "model": model, "max_len": max_tokens,
        "stream": False, "enable_thinking": thinking, "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        api.rstrip("/") + "/generate", data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data.get("generated_text", ""), time.time() - t0


def strip_reasoning(text):
    """Scoring looks at the answer, not at the chain of thought."""
    if text.startswith("> **Reasoning**"):
        lines = text.split("\n")
        i = 1
        while i < len(lines) and (lines[i].startswith(">") or not lines[i].strip()):
            i += 1
        return "\n".join(lines[i:]).strip()
    return text.strip()


# Subscripts and full-width forms are the same answer written differently. "H₂O"
# failed a check for "H2O" until this existed - a fault in the check, not the model.
_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹", "01234567890123456789")


def norm(s):
    s = (s or "").translate(_SUB)
    s = s.replace("×", "x").replace("＊", "*")
    return re.sub(r"\s+", " ", s).strip()


def check(case, answer):
    """Return (passed, why). Only the checks a case actually declares."""
    low = answer.lower()

    for kw in case.get("forbidden_keywords", []):
        if kw.lower() in low:
            return False, "contains forbidden %r" % kw

    expected = case.get("expected_answer")
    if expected is not None:
        if norm(expected).lower() not in norm(answer).lower():
            return False, "missing expected %r" % expected

    # A keyword may be a list of acceptable alternatives: "不必" and "多余" are both
    # right, and demanding one specific wording tests the checker's taste, not the model.
    missing = []
    for k in case.get("expected_keywords", []):
        options = k if isinstance(k, list) else [k]
        if not any(str(o).lower() in low for o in options):
            missing.append("|".join(str(o) for o in options))
    if missing:
        return False, "missing %s" % ", ".join(repr(m) for m in missing)

    for pat in case.get("must_match", []):
        if not re.search(pat, answer, re.I | re.S):
            return False, "no match for /%s/" % pat
    for pat in case.get("must_not_match", []):
        if re.search(pat, answer, re.I | re.S):
            return False, "matched /%s/" % pat

    if case.get("min_distinct_lines"):
        lines = [norm(l) for l in answer.split("\n") if norm(l)]
        if len(set(lines)) < case["min_distinct_lines"]:
            return False, "only %d distinct lines" % len(set(lines))

    limit = case.get("max_chars")
    if limit and len(answer) > limit:
        return False, "%d chars over limit %d" % (len(answer), limit)

    if case.get("min_new_tokens") and len(answer) < case["min_new_tokens"]:
        return False, "answer too short (%d)" % len(answer)

    # Degeneration: the failure mode v1 claimed to test and did not. A looping model
    # repeats a small set of fragments, so distinct 8-character windows collapse.
    ratio_limit = case.get("max_repeat_ratio")
    if ratio_limit and len(answer) >= 32:
        windows = [answer[i:i + 8] for i in range(0, len(answer) - 8)]
        if windows:
            ratio = len(set(windows)) / float(len(windows))
            if ratio < ratio_limit:
                return False, "repetitive (distinct ratio %.2f < %.2f)" % (ratio, ratio_limit)

    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="cases_v2.jsonl")
    ap.add_argument("--model", default="tfmf")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--key", default=DEFAULT_KEY)
    ap.add_argument("--thinking", action="store_true")
    # 0 so a published score can be reproduced. At the service default the same
    # case set scored 75.6% and then 85.4% on consecutive runs, which is not a
    # number anyone can quote.
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cases = [json.loads(l) for l in Path(args.cases).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        cases = cases[:args.limit]

    rows, by_cat = [], {}
    for n, case in enumerate(cases, 1):
        try:
            raw, secs = ask(args.api, args.key, args.model, case["user"],
                            case.get("max_new_tokens", 256), args.thinking, args.temperature)
            answer = strip_reasoning(raw)
            ok, why = check(case, answer)
        except Exception as exc:
            answer, secs, ok, why = "", 0.0, False, "%s: %s" % (type(exc).__name__, exc)
        cat = case.get("category", "?")
        by_cat.setdefault(cat, [0, 0])
        by_cat[cat][1] += 1
        if ok:
            by_cat[cat][0] += 1
        rows.append({"id": case["id"], "category": cat, "user": case["user"],
                     "answer": answer, "passed": ok, "why": why, "seconds": round(secs, 2)})
        print("  [%2d/%d] %-8s %-5s %s" % (n, len(cases), case["id"],
                                           "PASS" if ok else "FAIL", why), file=sys.stderr, flush=True)

    total = len(rows)
    passed = sum(1 for r in rows if r["passed"])
    summary = {"model": args.model, "thinking": args.thinking,
               "temperature": args.temperature, "total": total, "passed": passed,
               "score": round(100.0 * passed / total, 1) if total else 0.0,
               "by_category": {k: {"passed": v[0], "total": v[1],
                                   "score": round(100.0 * v[0] / v[1], 1) if v[1] else 0.0}
                               for k, v in sorted(by_cat.items())}}
    Path(args.out).write_text(json.dumps({"summary": summary, "results": rows},
                                         ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    for k, v in summary["by_category"].items():
        print("    %-14s %2d/%2d  %5.1f%%" % (k, v["passed"], v["total"], v["score"]))
    print("    %-14s %2d/%2d  %5.1f%%" % ("TOTAL", passed, total, summary["score"]))


if __name__ == "__main__":
    main()
