import requests
import json

GEN_PROMPT_TEMPLATE = """You are helping generate plausible compiler error messages written by student compilers.

You will be given the official compiler error message for a program. Write an alternative error message that:
- Describes the SAME root cause in your own natural wording.
- May simplify terms, reorder phrases, or add hints like line numbers, variable names, etc.
- Looks like it was printed by a student-built compiler (so it's okay if it's a bit rough or inconsistent).
- Must NOT claim to be an official or reference compiler.
- Must NOT copy the exact text verbatim.

Return ONLY the student's error message, nothing else.

OFFICIAL ERROR:
"{expected_error}"

STUDENT COMPILER MESSAGE:
"""

BATCH_JUDGE_JSON_TEMPLATE = """You are an automatic grader for a compiler course.

Task:
You will receive multiple pairs of compiler error messages:
  - EXPECTED_ERROR: the official/reference compiler error for a program.
  - STUDENT_ERROR: the message printed by a student's custom compiler on the same program.

For each pair, decide if the STUDENT_ERROR is an acceptable match for EXPECTED_ERROR.

Rules for "acceptable match":
- The STUDENT_ERROR must describe the SAME underlying problem (same root cause).
- Different wording is allowed. Synonyms are allowed. For example:
    - "token", "symbol", "operator", "character" can refer to the same thing.
    - "EOL" = "end of line".
    - "EOF" = "end of file".
    - "identifier not found" ≈ "variable is not defined".
    - "incompatible types" ≈ "expected integer but found string".
- Extra details (line number, column, hints, variable names, etc.) are allowed.
- It is acceptable if the student message explains the expected thing instead of repeating the wrong thing.
- It is NOT acceptable if the STUDENT_ERROR points to a different root cause.

Some examples (for your understanding, you do not need to repeat them):

Example 1 (ACCEPTABLE, True):
  EXPECTED_ERROR: "syntax error: unexpected ';' before ')'"
  STUDENT_ERROR: "parse error: found ';' where a ')' was expected"
  Reason: same location and same underlying syntax issue.

Example 2 (ACCEPTABLE, True):
  EXPECTED_ERROR: "undefined variable x"
  STUDENT_ERROR: "identifier 'x' not declared in this scope"
  Reason: both say x is not defined.

Example 3 (NOT ACCEPTABLE, False):
  EXPECTED_ERROR: "type mismatch: expected int but got string"
  STUDENT_ERROR: "missing semicolon at end of line"
  Reason: the causes are completely different.

Example 4 (ACCEPTABLE, True):
  EXPECTED_ERROR: "unexpected end of file while parsing expression"
  STUDENT_ERROR: "parser reached EOF while reading an expression"
  Reason: both describe EOF in the middle of parsing an expression.

Output format:
You MUST output a single JSON array.
Each element MUST be an object with two keys:
  - "index": an integer index of the pair, starting from 1
  - "correct": a boolean (true or false)

The JSON must look like:
[
  { "index": 1, "correct": true },
  { "index": 2, "correct": false },
  ...
]

Do NOT output anything before or after the JSON array.
Do NOT include comments, explanations, or extra keys.

Here are the pairs:

{pairs_block}
"""


JUDGE_FEWSHOT_TEMPLATE = """You are an automatic grader for a compiler course.

Task:
You will receive two compiler error messages:
1. EXPECTED_ERROR: the reference/official compiler's message for a specific program.
2. STUDENT_ERROR: the message printed by a student's custom compiler on the same program.

Decide if the STUDENT_ERROR is an acceptable match for EXPECTED_ERROR.

Rules for "acceptable match":
- The STUDENT_ERROR must describe the SAME underlying problem (same root cause).
- Different wording is allowed. Synonyms are allowed.
  Examples:
    - "token", "symbol", "operator", "character" can mean the same thing.
    - "EOL" means "end of line".
    - "EOF" means "end of file".
    - "identifier not found" means "variable not defined".
    - "incompatible types" means "expected integer but found string".
- Extra detail like line numbers, hints, or variable names is allowed.
- It's still acceptable if the student message explains the expected thing instead of repeating the wrong thing.
- It is NOT acceptable if the STUDENT_ERROR points to a different root cause.

Output ONLY one token: True or False

Examples (study them carefully):

Example 1:
EXPECTED_ERROR: "Invalid token ,"
STUDENT_ERROR: "Line 5: Unknown symbol ',' found. Did you mean to use a different operator?"
ANSWER: True
# Reason: "token" vs "symbol" is the same idea: the comma is not allowed.

Example 2:
EXPECTED_ERROR: "Unexpected token EOL"
STUDENT_ERROR: "Line 7: Unexpected end of line in expression. Make sure all operators have matching operands."
ANSWER: True
# Reason: "EOL" and "end of line" describe the same parse stop.

Example 3:
EXPECTED_ERROR: "Unexpected token EOF"
STUDENT_ERROR: "Line 10: Unexpected end of file. Did you forget to close that parenthesis?"
ANSWER: True
# Reason: "EOF" and "end of file" are equivalent. Both complain that input ended too early / something not closed.

Example 4:
EXPECTED_ERROR: "Unexpected token MULT"
STUDENT_ERROR: "Line 10: Unknown symbol '*'. Did you mean to use a different operator?"
ANSWER: True
# Reason: "MULT" is the '*' operator. Student calls it 'symbol *'. Same root cause: '*' appeared where it shouldn't.

Example 5:
EXPECTED_ERROR: "Unexpected token IDEN"
STUDENT_ERROR: "Line 5: Unexpected identifier 'myVar' found here. Did you mean to use a number or a string instead?"
ANSWER: True
# Reason: "token IDEN" == "an identifier here". Student just rephrased it and gave an example.

Example 6:
EXPECTED_ERROR: "Identifier not found"
STUDENT_ERROR: "On line 10, the variable 'count' was used but not defined. Make sure you declare your variables before using them."
ANSWER: True
# Reason: "identifier not found" == "variable was used but not declared".

Example 7:
EXPECTED_ERROR: "Incompatible Type"
STUDENT_ERROR: "Error: Expected integer but found string on line 10 near var age."
ANSWER: True
# Reason: "incompatible type" == "expected int but got string". Same type mismatch root cause.

Example 8:
EXPECTED_ERROR: "Missing OPEN_BRA"
STUDENT_ERROR: "On line 7, you forgot to open a parenthesis before your function call. Remember to add '('."
ANSWER: True
# Reason: "OPEN_BRA" / "open bracket/parenthesis missing" are the same complaint.

Example 9:
EXPECTED_ERROR: "Unexpected token EOL"
STUDENT_ERROR: "Line 5: Expected a value after 'x='"
ANSWER: True
# Reason: We hit end of line where a value should be. That matches 'Unexpected token EOL'.

Example 10:
EXPECTED_ERROR: "Unexpected token EOL"
STUDENT_ERROR: "student compiler: Unexpected token EOF"
ANSWER: False
# Reason: 'EOL' (end of line) and 'EOF' (end of file) are NOT always the same problem.

Now judge this pair:

EXPECTED_ERROR:
"{expected_error}"

STUDENT_ERROR:
"{student_error}"

ANSWER:
"""

BATCH_JUDGE_TEMPLATE = """You are an automatic grader for a compiler course.

Task:
You will receive multiple pairs of compiler error messages:
1. EXPECTED_ERROR: the reference/official compiler's message for a specific program.
2. STUDENT_ERROR: the message printed by a student's custom compiler on the same program.

For each pair, decide if the STUDENT_ERROR is an acceptable match for EXPECTED_ERROR.

Rules for "acceptable match":
- The STUDENT_ERROR must describe the SAME underlying problem (same root cause).
- Different wording is allowed. Synonyms are allowed.
  Examples:
    - "token", "symbol", "operator", "character" can mean the same thing.
    - "EOL" means "end of line".
    - "EOF" means "end of file".
    - "identifier not found" means "variable not defined".
    - "incompatible types" means "expected integer but found string".
- Extra detail like line numbers, hints, or variable names is allowed.
- It's still acceptable if the student message explains the expected thing instead of repeating the wrong thing.
- It is NOT acceptable if the STUDENT_ERROR points to a different root cause.

Your output format:
- Return one line per pair, in order.
- Each line must be exactly either: True or False (case-insensitive is okay).

Here are the pairs:

{pairs_block}

ANSWERS:
"""

def call_ollama(prompt: str,
                model: str = "qwen2.5:3b-instruct",
                temperature: float = 0.0,
                max_tokens: int = 32) -> str:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
        "stream": True
    }
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        full = []
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                full.append(data["response"])
            if data.get("done", False):
                break
        return "".join(full).strip()

def normalize_bool(model_raw: str) -> bool:
    first = model_raw.strip().split()[0]
    return first.lower().startswith("true")
