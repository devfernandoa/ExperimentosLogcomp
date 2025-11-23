import zipfile
import yaml
import json

INPUT_ZIP = "testslogcomp.zip"
OUT_PATH = "gold.jsonl"

def main():
    gold_examples = []

    with zipfile.ZipFile(INPUT_ZIP, "r") as zf:
        for member in zf.namelist():
            # We only expect YAML test definition files in this zip
            if not member.lower().endswith(".yaml"):
                continue
            with zf.open(member) as f:
                data = yaml.safe_load(f)

            # Each YAML file is a list of test cases
            for idx, case in enumerate(data):
                # We only care about cases where the program SHOULD raise an error
                # ('exception' == True), and we grab the expected error message
                if case.get("exception"):
                    expected_error = (case.get("output") or "").strip()

                    # Build a stable id like "v1.0.yaml::12"
                    test_id = f"{member}::{idx}"

                    gold_examples.append({
                        "test_id": test_id,
                        "expected_error": expected_error
                    })

    # Write JSONL
    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for row in gold_examples:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {len(gold_examples)} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()
