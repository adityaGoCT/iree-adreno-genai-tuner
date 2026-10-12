"""Summarize bench JSONL files: per (tag, schedule, ctx) medians over prompt kinds + speedups.

  summarize.py models/llama32-3b-q4/bench/baseline.jsonl models/llama32-3b-q4/bench/tuned.jsonl ...
Prints a markdown table; --json writes the aggregated rows for plotting.
"""

import argparse
import collections
import json
import pathlib
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=pathlib.Path)
    ap.add_argument("--json", type=pathlib.Path)
    a = ap.parse_args()
    groups = collections.defaultdict(list)
    for f in a.files:
        for line in f.read_text().splitlines():
            r = json.loads(line)
            groups[(r["tag"], r["schedule"], r["ctx"])].append(r)
    rows = []
    for (tag, sched, ctx), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        rows.append(dict(tag=tag, schedule=sched, ctx=ctx, n=len(rs),
                         prefill_tok_s=statistics.median(r["prefill_tok_s"] for r in rs),
                         ttft_s=statistics.median(r["ttft_ms"] for r in rs) / 1e3,
                         decode_tok_s=statistics.median(r["decode_tok_s"] for r in rs)))
    base = {(r["ctx"]): r for r in rows if r is rows[0] or (r["tag"], r["schedule"]) == (rows[0]["tag"], rows[0]["schedule"])}
    print("| config | ctx | prefill tok/s | TTFT s | decode tok/s | prefill × | decode × |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        b = base.get(r["ctx"])
        sp = r["prefill_tok_s"] / b["prefill_tok_s"] if b else float("nan")
        sd = r["decode_tok_s"] / b["decode_tok_s"] if b else float("nan")
        print(f"| {r['tag']}/{r['schedule']} | {r['ctx']} | {r['prefill_tok_s']:.1f} | {r['ttft_s']:.2f} | "
              f"{r['decode_tok_s']:.2f} | {sp:.2f} | {sd:.2f} |")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
