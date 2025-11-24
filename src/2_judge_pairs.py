import json
import time
from pathlib import Path
from prompts import JUDGE_FEWSHOT_TEMPLATE, call_ollama, normalize_bool

SYNTH_PATH = Path("../data/synthetic.jsonl")
JUDGE_PATH = Path("../data/judgments.jsonl")

def main():
    # Load pairs
    pairs = []
    with SYNTH_PATH.open(encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    judged_rows = []

    # If there's any one-time "model load", measure it here.
    # For an API-style model like call_ollama, there's not really a load step
    # unless you want to force-load the model with a dummy call.
    # We'll keep both numbers for consistency with the sentence-transformer report.
    #
    # 1) "Model load time": optional warmup call with a no-op prompt.
    # 2) "Total inference time": sum of real pair evaluations.
    #
    # If you don't want to incur an extra request, you can skip warmup and just set load_secs = 0.
    warmup_prompt = (
        "You are a health check. Reply with True.\nANSWER:\n"
    )
    t_load_start = time.time()
    _ = call_ollama(
        prompt=warmup_prompt,
        temperature=0.0,
        max_tokens=4
    )
    t_load_end = time.time()
    load_secs = t_load_end - t_load_start

    # Now measure actual judging time across all pairs
    t_infer_start = time.time()
    for row in pairs:
        prompt = JUDGE_FEWSHOT_TEMPLATE.format(
            expected_error=row["expected_error"],
            student_error=row["student_error"]
        )

        t0 = time.time()
        raw = call_ollama(
            prompt=prompt,
            temperature=0.0,
            max_tokens=8
        )
        t1 = time.time()

        model_bool = normalize_bool(raw)

        judged_rows.append({
            **row,
            "model_output": raw,
            "model_bool": model_bool,
            "latency_sec": t1 - t0  # per-example latency if you want to keep it
        })
    t_infer_end = time.time()

    infer_secs_total = t_infer_end - t_infer_start
    avg_ms_per_pair = (infer_secs_total / len(pairs)) * 1000.0

    # Write out judgments.jsonl (same as before)
    with JUDGE_PATH.open("w", encoding="utf-8") as f:
        for r in judged_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print timing summary at the end
    print(f"Wrote {len(judged_rows)} judged pairs to {JUDGE_PATH}")
    print()
    print("=== TIMING (Qwen judge) ===")
    print(f"Model warmup/load time (s): {load_secs:.3f}")
    print(f"Total inference time (s): {infer_secs_total:.3f}")
    print(f"Avg inference time per pair (ms): {avg_ms_per_pair:.3f}")

    # (Optional) also show distribution of per-example latency,
    # since Ollama is sequential
    latencies = [r["latency_sec"] for r in judged_rows]
    p50 = sorted(latencies)[len(latencies)//2]
    p95 = sorted(latencies)[int(len(latencies)*0.95)]
    print(f"P50 single-call latency (ms): {p50*1000.0:.3f}")
    print(f"P95 single-call latency (ms): {p95*1000.0:.3f}")

if __name__ == "__main__":
    main()
