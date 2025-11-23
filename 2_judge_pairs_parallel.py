import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from prompts import JUDGE_FEWSHOT_TEMPLATE, call_ollama, normalize_bool

SYNTH_PATH = Path("synthetic.jsonl")
JUDGE_PATH = Path("judgments.jsonl")

# You can tune this (8, 12, 16, ...)
MAX_WORKERS = 16


def load_pairs():
    pairs = []
    with SYNTH_PATH.open(encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def judge_one(row):
    prompt = JUDGE_FEWSHOT_TEMPLATE.format(
        expected_error=row["expected_error"],
        student_error=row["student_error"],
    )

    t0 = time.time()
    raw = call_ollama(
        prompt=prompt,
        temperature=0.0,
        max_tokens=8,
    )
    t1 = time.time()

    model_bool = normalize_bool(raw)

    return {
        **row,
        "model_output": raw,
        "model_bool": model_bool,
        "latency_sec": t1 - t0,
    }


def main():
    pairs = load_pairs()
    judged_rows = []

    # Warmup (same as original)
    warmup_prompt = "You are a health check. Reply with True.\nANSWER:\n"
    t_load_start = time.time()
    _ = call_ollama(
        prompt=warmup_prompt,
        temperature=0.0,
        max_tokens=4,
    )
    t_load_end = time.time()
    load_secs = t_load_end - t_load_start

    # Parallel inference
    t_infer_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(judge_one, row) for row in pairs]
        for fut in as_completed(futures):
            judged_rows.append(fut.result())
    t_infer_end = time.time()

    infer_secs_total = t_infer_end - t_infer_start
    avg_ms_per_pair = (infer_secs_total / len(judged_rows)) * 1000.0

    # Write results
    with JUDGE_PATH.open("w", encoding="utf-8") as f:
        for r in judged_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(judged_rows)} judged pairs to {JUDGE_PATH}")
    print()
    print("=== TIMING (Qwen judge, parallel) ===")
    print(f"Model warmup/load time (s): {load_secs:.3f}")
    print(f"Total inference time (s): {infer_secs_total:.3f}")
    print(f"Avg inference time per pair (ms): {avg_ms_per_pair:.3f}")

    latencies = [r["latency_sec"] for r in judged_rows]
    if latencies:
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies_sorted) // 2]
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
        print(f"P50 single-call latency (ms): {p50 * 1000.0:.3f}")
        print(f"P95 single-call latency (ms): {p95 * 1000.0:.3f}")


if __name__ == "__main__":
    main()
