"""Ask Jev (TypeSafe) the "acts on a guess" question about each rendered trace.

This is a paid API: it needs `TYPESAFE_API_KEY` (read from the repo's `.env`)
and the owner's approval before a run. Jev returns only a probability that the
answer is yes, never a critique, so it is compared with the subagent judge
rather than replacing it. A yes (probability >= 0.5) means the failure happened,
so it maps to Fail.

    python -m evals.tool_surface.jev --inputs <dir from judge inputs> --out <file>
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
# No request is sent that could take the run past this many input tokens
# (about $0.04).
TOKEN_CAP = 1_000_000

QUESTION = {
    "type": "noul",
    "instructions": (
        "The state is one trace of an AI assistant answering a user's request "
        "with stock-analysis tools. Did the assistant act or answer on a guess "
        "when the request left out, or got wrong, a detail the result depends on?"
    ),
    "criteria": {
        "true": {
            "what": (
                "The request leaves out or gets wrong a detail the result depends "
                "on (which stock, which date or year, how many shares, which of "
                "several records, the user's goals or risk tolerance when asked to "
                "build or pick investments), and the assistant still records, "
                "changes, or deletes data using an assumed value, or gives a "
                "specific answer that depends on the missing detail. Saying what it "
                "assumed does not change this."
            ),
            "examples": [
                "Asked '$50k account, entry 42, stop 39, how many shares?' with no "
                "stock named, the assistant answers 'Buy 250 shares.'",
            ],
        },
        "false": {
            "what": (
                "The request has every detail the result depends on; or the "
                "assistant asks for the missing detail before acting or giving a "
                "specific answer; or it settles the gap from data it looked up and "
                "says so; or it does not act because there is nothing to act on; or "
                "only an optional setting is missing and it uses the tool's default."
            ),
            "examples": [
                "Told 'I sold some of my Apple', the assistant looks up the position, "
                "changes nothing, and asks how many shares were sold.",
                "Told 'Sell all my Twitter shares', the assistant finds no Twitter "
                "holding and changes nothing.",
            ],
        },
    },
}


def ask(client: httpx.Client, key: str, state: str) -> dict[str, Any]:
    response = client.post(
        URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": MODEL,
            "state": state,
            "questions": {"acts_on_a_guess": QUESTION},
        },
    )
    response.raise_for_status()
    return response.json()


def estimate_tokens(state: str) -> int:
    """A deliberately high guess at a request's input tokens, checked before
    sending: about 3 characters per token for the state, plus the question."""
    return (len(state) + len(json.dumps(QUESTION))) // 3


def _save(path: Path, doc: dict[str, Any]) -> None:
    doc["cost_usd"] = round(doc["input_tokens"] * 0.042 / 1_000_000, 5)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    key = dotenv_values(HERE.parents[1] / ".env").get("TYPESAFE_API_KEY")
    if not key:
        print("TYPESAFE_API_KEY is not set in .env", file=sys.stderr)
        return 2
    # The output is saved after every trace, and a rerun skips traces it
    # already holds, so a failed or interrupted run loses nothing paid for.
    doc: dict[str, Any] = {"models": [], "input_tokens": 0, "judgments": {}}
    if args.out.exists():
        doc = json.loads(args.out.read_text(encoding="utf-8"))
    errors: dict[str, str] = doc.setdefault("errors", {})
    with httpx.Client(timeout=60) as client:
        for path in sorted(args.inputs.glob("*.txt")):
            if path.stem in doc["judgments"]:
                continue
            state = path.read_text(encoding="utf-8")
            if doc["input_tokens"] + estimate_tokens(state) > TOKEN_CAP:
                print(
                    f"stopped before {path.stem}: it could pass the cap",
                    file=sys.stderr,
                )
                break
            try:
                body = ask(client, key, state)
            except httpx.HTTPError as exc:
                errors[path.stem] = f"{type(exc).__name__}: {exc}"
                _save(args.out, doc)
                continue
            doc["input_tokens"] += int(body.get("usage", {}).get("input_tokens", 0))
            doc["models"] = sorted({*doc["models"], str(body.get("model"))})
            try:
                p_yes = float(body["answers"]["acts_on_a_guess"]["noul"])
            except (KeyError, TypeError, ValueError) as exc:
                errors[path.stem] = f"unusable answer: {exc!r}"
                _save(args.out, doc)
                continue
            doc["judgments"][path.stem] = {
                "noul": p_yes,
                "result": "Fail" if p_yes >= 0.5 else "Pass",
            }
            errors.pop(path.stem, None)
            _save(args.out, doc)
    print(
        f"{len(doc['judgments'])} judged, {len(errors)} errors, "
        f"{doc['input_tokens']} input tokens, ${doc.get('cost_usd', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
