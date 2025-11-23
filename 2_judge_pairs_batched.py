import json
import time
from pathlib import Path
from prompts import BATCH_JUDGE_TEMPLATE, call_ollama, normalize_bool

SYNTH_PATH = Path("synthetic.jsonl")
JUDGE_PATH = Path("judgments.jsonl")

BATCH_SIZE = 16  # tweak this up or down depending on context length / speed

def load_pairs():
    pairs = []
    with SYNTH_PATH.open(encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs

def build_batch_prompt(batch_rows):
    parts = []
    for i, row in enumerate(batch_rows, start=1):
        parts.append(
            f"Pair {i}:\n"
            f"EXPECTED_ERROR:\n{row['expected_error']}\n\n"
            f"STUDENT_ERROR:\n{row['student_error']}\n"
        )
    pairs_block = "\n".join(parts)
    return BATCH_JUDGE_TEMPLATE.format(pairs_block=pairs_block)

def parse_batch_output(raw, batch_len):
    # Primary: assume model returns one True/False per line
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    answers = []

    if len(lines) >= batch_len:
        answers = lines[:batch_len]
    else:
        # Fallback: pull True/False-like tokens from entire text
        tokens = raw.replace("\n", " ").split()
        for t in tokens:
            tl = t.lower()
            if tl.startswith("true") or tl.startswith("false"):
                answers.append(t)
                if len(answers) == batch_len:
                    break

    if len(answers) != batch_len:
        raise ValueError(
            f"Could not parse {batch_len} answers from model output:\n{raw}"
        )

    return answers

def main():
    pairs = load_pairs()

    judged_rows = []

    # Optional warmup call, same as your original script
    warmup_prompt = "You are a health check. Reply with True.\nANSWER:\n"
    t_load_start = time.time()
    _ = call_ollama(
        prompt=warmup_prompt,
        temperature=0.0,
        max_tokens=4
    )
    t_load_end = time.time()
    load_secs = t_load_end - t_load_start

    # Measure batched inference time
    t_infer_start = time.time()
    batch_latencies = []

    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i : i + BATCH_SIZE]
        prompt = build_batch_prompt(batch)

        t0 = time.time()
        raw = call_ollama(
            prompt=prompt,
            temperature=0.0,
            max_tokens=8 * len(batch)  # a few tokens per answer
        )
        t1 = time.time()
        batch_latencies.append(t1 - t0)

        answers = parse_batch_output(raw, len(batch))

        for row, ans_raw in zip(batch, answers):
            model_bool = normalize_bool(ans_raw)
            judged_rows.append({
                **row,
                "model_output": ans_raw,
                "model_bool": model_bool,
                # Approximate per-example latency inside this batch
                "latency_sec": (t1 - t0) / len(batch)
            })

    t_infer_end = time.time()
    infer_secs_total = t_infer_end - t_infer_start
    avg_ms_per_pair = (infer_secs_total / len(pairs)) * 1000.0

    # Write out judgments.jsonl with the same schema your eval script expects
    with JUDGE_PATH.open("w", encoding="utf-8") as f:
        for r in judged_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Timing report (similar to original)
    print(f"Wrote {len(judged_rows)} judged pairs to {JUDGE_PATH}")
    print()
    print("=== TIMING (Qwen judge, batched) ===")
    print(f"Model warmup/load time (s): {load_secs:.3f}")
    print(f"Total inference time (s): {infer_secs_total:.3f}")
    print(f"Avg inference time per pair (ms): {avg_ms_per_pair:.3f}")

    if batch_latencies:
        sorted_lat = sorted(batch_latencies)
        p50 = sorted_lat[len(sorted_lat)//2]
        p95 = sorted_lat[int(len(sorted_lat)*0.95)]
        print(f"P50 batch-call latency (ms): {p50*1000.0:.3f}")
        print(f"P95 batch-call latency (ms): {p95*1000.0:.3f}")

if __name__ == "__main__":
    main()
