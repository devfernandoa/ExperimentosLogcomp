import json
import random
from pathlib import Path
from prompts import GEN_PROMPT_TEMPLATE, call_ollama

GOLD_PATH = Path("gold.jsonl")
SYNTH_PATH = Path("synthetic.jsonl")

def main():
    random.seed(1337)

    # load gold
    gold_examples = []
    with GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            gold_examples.append(json.loads(line))

    synthetic_rows = []

    # 1. Positive pairs (label=True)
    for ex in gold_examples:
        gen_prompt = GEN_PROMPT_TEMPLATE.format(
            expected_error=ex["expected_error"]
        )
        student_msg = call_ollama(
            prompt=gen_prompt,
            temperature=0.7,    # a bit creative for paraphrasing
            max_tokens=64
        ).strip()

        synthetic_rows.append({
            "test_id": ex["test_id"],
            "expected_error": ex["expected_error"],
            "student_error": student_msg,
            "label": True
        })

    # 2. Negative pairs (label=False)
    # We'll shuffle and pair mismatching errors.
    if len(gold_examples) > 1:
        shuffled = gold_examples[:]
        random.shuffle(shuffled)

        for ex, wrong in zip(gold_examples, shuffled):
            if ex["test_id"] == wrong["test_id"]:
                continue
            # We'll just reuse wrong.expected_error as if a student's compiler
            # said something completely different.
            synthetic_rows.append({
                "test_id": ex["test_id"] + "_neg",
                "expected_error": ex["expected_error"],
                "student_error": "student compiler: " + wrong["expected_error"],
                "label": False
            })

    with SYNTH_PATH.open("w", encoding="utf-8") as f:
        for row in synthetic_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(synthetic_rows)} pairs to {SYNTH_PATH}")

if __name__ == "__main__":
    main()
