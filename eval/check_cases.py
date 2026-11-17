"""Validate the evaluation cases: uv run python -m eval.check_cases [path]"""

import json
import sys
from pathlib import Path

from eval.cases import coverage_problems, load_cases

DEFAULT_PATH = Path(__file__).resolve().parent / "cases.json"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_PATH
    if not path.exists():
        print(f"{path} not found.")
        return 1
    try:
        cases = load_cases(path)
    except (ValueError, json.JSONDecodeError) as error:
        print(f"{path} is not valid:\n{error}")
        return 1
    problems = coverage_problems(cases)
    for problem in problems:
        print(f"- {problem}")
    if problems:
        return 1
    print(f"OK: {len(cases)} cases in {path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
