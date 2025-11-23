import json
import time
from pathlib import Path
from prompts import BATCH_JUDGE_JSON_TEMPLATE, call_ollama, normalize_bool

SYNTH_PATH = Path("synthetic.jsonl")
JUDGE_PATH = Path("judgments.jsonl")

# Tradeoff: bigger batch => faster, but longer prompt
BATCH_SIZE = 8

def load_pairs():
    pairs = []
    with SYNTH_PATH.open(encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs

def build_pairs_block(batch_rows):
    """
    Format as a numbered list that we refer to via `index` in JSON.
    """
    parts = []
    for i, row in enumerate(batch_rows, start=1):
        parts.append(
            f"Pair {i}:\n"
            f"EXPECTED_ERROR:\n{row['expected_error']}\n\n"
            f"STUDENT_ERROR:\n{row['student_error']}\n"
        )
    return "\n".join(parts)

def call_batch_judge(batch_rows):
    pairs_block = build_pairs_block(batch_rows)
    # Use .replace instead of .format to avoid brace issues
    prompt = BATCH_JUDGE_JSON_TEMPLATE.replace("{pairs_block}", pairs_block)

    raw = call_ollama(
        prompt=prompt,
        temperature=0.0,
        # Plenty of room for JSON; still cheap compared to 8 separate calls
        max_tokens=512,
    )
    raw = raw.strip()

    # Defensive: try to isolate JSON array if the model wraps it
    try:
        if not raw.lstrip().startswith("["):
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                raw = raw[start:end + 1]

        data = json.loads(raw)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from model output:\n{raw}\nError: {e}")


    if not isinstance(data, list):
        raise ValueError(f"Model JSON is not a list: {data!r}")

    # Map index -> raw bool-like value
    answers_by_index = {}
    for obj in data:
        if not isinstance(obj, dict):
            continue
        idx = obj.get("index")
        correct = obj.get("correct")
        if isinstance(idx, int):
            answers_by_index[idx] = correct

    if len(answers_by_index) != len(batch_rows):
        raise ValueError(
            f"Expected {len(batch_rows)} answers, got {len(answers_by_index)}. "
            f"Model JSON: {data!r}"
        )

    # Return answers in order 1..len(batch_rows)
    return [answers_by_index[i] for i in range(1, len(batch_rows) + 1)]

def main():
    pairs = load_pairs()
    judged_rows = []

    # Warmup / load time (same as your original script)
    warmup_prompt = "You are a health check. Reply with True.\nANSWER:\n"
    t_load_start = time.time()
    _ = call_ollama(
        prompt=warmup_prompt,
        temperature=0.0,
        max_tokens=4,
    )
    t_load_end = time.time()
    load_secs = t_load_end - t_load_start

    t_infer_start = time.time()
    batch_latencies = []

    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start : start + BATCH_SIZE]

        t0 = time.time()
        batch_answers = call_batch_judge(batch)
        t1 = time.time()
        batch_latencies.append(t1 - t0)

        for row, ans in zip(batch, batch_answers):
            # normalize_bool expects a string; we can feed "True"/"False"
            # but if model gives us a real boolean, handle that too.
            if isinstance(ans, bool):
                ans_raw = "True" if ans else "False"
            else:
                ans_raw = str(ans)

            model_bool = normalize_bool(ans_raw)

            judged_rows.append({
                **row,
                "model_output": ans_raw,
                "model_bool": model_bool,
                "latency_sec": (t1 - t0) / len(batch),
            })

    t_infer_end = time.time()
    infer_secs_total = t_infer_end - t_infer_start
    avg_ms_per_pair = (infer_secs_total / len(judged_rows)) * 1000.0

    with JUDGE_PATH.open("w", encoding="utf-8") as f:
        for r in judged_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(judged_rows)} judged pairs to {JUDGE_PATH}")
    print()
    print("=== TIMING (Qwen judge, batched v2) ===")
    print(f"Model warmup/load time (s): {load_secs:.3f}")
    print(f"Total inference time (s): {infer_secs_total:.3f}")
    print(f"Avg inference time per pair (ms): {avg_ms_per_pair:.3f}")

    if batch_latencies:
        sorted_lat = sorted(batch_latencies)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p95 = sorted_lat[int(len(sorted_lat) * 0.95)]
        print(f"P50 batch-call latency (ms): {p50 * 1000.0:.3f}")
        print(f"P95 batch-call latency (ms): {p95 * 1000.0:.3f}")

if __name__ == "__main__":
    main()
