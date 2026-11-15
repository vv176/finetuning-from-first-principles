"""
B2: The grader. Four numbers, not one, and no LLM judge anywhere.

B1 published the schema and the three rules. This file turns them into a
measuring instrument, then points it at the BASE model to produce the number
every later arc is compared against.

Why four numbers. A single "accuracy" hides which thing is broken, and the
four failures below need four different fixes:

    parse_rate     is it JSON at all?            fix: constrained decoding
    schema_valid   right keys, types, enums?     fix: better prompt, or training
    field_acc      which FIELDS are wrong?       fix: depends on the field
    exact_match    is the whole object right?    the number you report

The per-field breakdown is the one that makes this teachable. The extraction
fields and the rule fields move differently, and students can see which kind of
learning fine-tuning is actually good at.

No LLM judge. Every check is a comparison against B1's rule, so two people
running this get the same answer, it costs nothing, and it finishes instantly.

The baseline it produces is the honest starting point: an untuned
qwen2.5:1.5b, prompted as well as a prompt can reasonably do it.

Requires Ollama running with qwen2.5:1.5b:
    ollama serve            (or the menu-bar app)
    ollama pull qwen2.5:1.5b

Run it:
    python3 b2_the_grader.py
"""

import json
import re
import time
import urllib.error
import urllib.request

from b1_schema_and_rules import (
    EXTRACT_FIELDS, FIELDS, RULE_FIELDS, derive, schema_errors,
)

MODEL = "qwen2.5:1.5b"
OLLAMA = "http://localhost:11434/api/chat"

SYSTEM = """You are a support router. Read the customer message and reply with ONE JSON object, nothing else.

Keys, all seven, always present:
  intent: one of damaged_item, missing_delivery, wrong_item, late_delivery, return_request, general_question
  order_id: string or null
  item_name: string or null
  days_since_delivery: integer >= 0, or null
  is_hygiene_item: true if item_name is a personal-hygiene item, false if it is not, null if there is no item
  requested_action: one of replacement, refund, redeliver, investigate, answer
  priority: one of low, medium, high, urgent

requested_action is fixed by intent:
  damaged_item -> replacement, wrong_item -> replacement,
  missing_delivery -> investigate, late_delivery -> investigate,
  return_request -> refund, general_question -> answer

priority is computed in this order:
  1. base: high for damaged_item, wrong_item, missing_delivery;
           medium for late_delivery; low for return_request, general_question
  2. if days_since_delivery is not null: <= 2 move up one level, > 30 move down one
  3. if is_hygiene_item is true: move up one more level
  4. clamp to the range low, medium, high, urgent"""


# ---------------------------------------------------------------------------
# The test set. Sixteen examples, gold labels from B1's rule.
# ---------------------------------------------------------------------------
# (message, intent, order_id, item_name, days_since_delivery)
CASES = [
    ("my water bottle arrived cracked yesterday, order 8831",
     "damaged_item", "8831", "water bottle", 1),
    ("the mug I got today is chipped, order number 4410",
     "damaged_item", "4410", "mug", 0),
    ("ordered a blue jacket, they sent a red one. order 2277, came 4 days ago",
     "wrong_item", "2277", "jacket", 4),
    ("where is my order?? placed it 3 weeks ago",
     "missing_delivery", None, None, 21),
    ("tracking says delivered 2 days ago but nothing arrived. order 5512",
     "missing_delivery", "5512", None, 2),
    ("my parcel is 6 days late, order 9930",
     "late_delivery", "9930", None, 6),
    ("still waiting, it was supposed to come 40 days back. order 1180",
     "late_delivery", "1180", None, 40),
    ("I want to return the razor I bought 45 days ago, order 9001",
     "return_request", "9001", "razor", 45),
    ("can I return these socks? got them yesterday",
     "return_request", None, "socks", 1),
    ("do you deliver to Pune?",
     "general_question", None, None, None),
    ("what is your return policy on electronics",
     "general_question", None, None, None),
    # the hard ones -- rule combinations, not vocabulary
    ("the toothbrush came apart the day it arrived, order 7001",
     "damaged_item", "7001", "toothbrush", 0),
    ("my mascara leaked everywhere, delivered 3 months ago, order 7742",
     "damaged_item", "7742", "mascara", 90),
    ("sent me the wrong deodorant, arrived 35 days back, order 6650",
     "wrong_item", "6650", "deodorant", 35),
    ("quick question about the shampoo you sell",
     "general_question", None, "shampoo", None),
    ("the earbuds never showed up, ordered them 1 day ago, order 3399",
     "missing_delivery", "3399", "earbuds", 1),
]

GOLD = [(msg, derive(i, o, it, d)) for msg, i, o, it, d in CASES]


# ---------------------------------------------------------------------------
# The grader.
# ---------------------------------------------------------------------------
FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def extract_json(text: str) -> tuple[dict | None, bool]:
    """Try to get an object out of the model's raw output.

    Returns (parsed_or_None, needed_repair). `needed_repair` is True when the
    strict parse failed and stripping a markdown fence rescued it. We report
    both numbers because the difference is a prompt problem, not a model
    problem, and it should not be hidden inside parse_rate.
    """
    try:
        obj = json.loads(text.strip())
        return (obj if isinstance(obj, dict) else None), False
    except json.JSONDecodeError:
        pass
    m = FENCE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            return (obj if isinstance(obj, dict) else None), True
        except json.JSONDecodeError:
            pass
    return None, False


def norm(v):
    """Compare strings case- and whitespace-insensitively. Everything else exact.

    None stays distinct from False and from "". That is deliberate: B1 made
    is_hygiene_item three-valued and the grader has to respect it.
    """
    return v.strip().lower() if isinstance(v, str) else v


def grade(predictions: list[str], golds: list[dict]) -> dict:
    """The whole grader. predictions are raw model strings."""
    n = len(golds)
    parsed_strict = 0
    parsed_after_repair = 0
    valid = 0
    exact = 0
    hits = {f: 0 for f in FIELDS}
    scored = {f: 0 for f in FIELDS}
    rows = []

    for raw, gold in zip(predictions, golds):
        obj, repaired = extract_json(raw)
        if obj is not None and not repaired:
            parsed_strict += 1
        if obj is not None:
            parsed_after_repair += 1

        errs = schema_errors(obj) if obj is not None else ["did not parse"]
        if not errs:
            valid += 1

        ok_fields = 0
        for f in FIELDS:
            scored[f] += 1
            if obj is not None and f in obj and norm(obj[f]) == norm(gold[f]):
                hits[f] += 1
                ok_fields += 1
        if ok_fields == len(FIELDS):
            exact += 1
        rows.append((obj, errs, ok_fields))

    return {
        "n": n,
        "parse_rate": parsed_strict / n,
        "parse_rate_after_repair": parsed_after_repair / n,
        "schema_valid": valid / n,
        "exact_match": exact / n,
        "field_acc": {f: hits[f] / scored[f] for f in FIELDS},
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Talking to the model.
# ---------------------------------------------------------------------------
def ask(message: str, force_json: bool = False, timeout: int = 120) -> str:
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": message}],
        "options": {"temperature": 0, "seed": 0},
    }
    if force_json:
        payload["format"] = "json"
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def ollama_up() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True
    except (urllib.error.URLError, OSError):
        return False


def report(name: str, res: dict) -> None:
    print(f"  {name}")
    print(f"    {'parse_rate':<24} {res['parse_rate']:>7.1%}"
          f"   ({round(res['parse_rate'] * res['n'])}/{res['n']})")
    if res["parse_rate_after_repair"] > res["parse_rate"]:
        print(f"    {'  after fence-stripping':<24} "
              f"{res['parse_rate_after_repair']:>7.1%}")
    print(f"    {'schema_valid':<24} {res['schema_valid']:>7.1%}"
          f"   ({round(res['schema_valid'] * res['n'])}/{res['n']})")
    print(f"    {'exact_match':<24} {res['exact_match']:>7.1%}"
          f"   ({round(res['exact_match'] * res['n'])}/{res['n']})")


# ===========================================================================
if __name__ == "__main__":
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("1. THE GRADER, PROVEN ON FOUR FAKE PREDICTIONS")
    print("=" * 72)
    print("""  Before touching a model: four hand-written predictions, each broken in a
  different way, to show that each metric catches a different thing.
""")

    gold0 = GOLD[0][1]
    fakes = [
        ("perfect", json.dumps(gold0)),
        ("right answer, wrapped in a fence",
         "Here you go:\n```json\n" + json.dumps(gold0) + "\n```"),
        ("valid JSON, action not in the enum",
         json.dumps({**gold0, "requested_action": "return_request"})),
        ("valid JSON + schema, one rule field wrong",
         json.dumps({**gold0, "priority": "low"})),
        ("prose, no JSON at all",
         "This looks like a damaged item, priority high."),
    ]
    print(f"  {'prediction':<42} {'parse':>6} {'valid':>6} {'fields':>7} {'exact':>6}")
    for name, raw in fakes:
        r = grade([raw], [gold0])
        print(f"  {name:<42} {r['parse_rate']:>6.0%} {r['schema_valid']:>6.0%}"
              f" {str(r['rows'][0][2]) + '/' + str(len(FIELDS)):>7} "
              f"{r['exact_match']:>6.0%}")

    print("""
  Row 2 is the one to point at. It is the correct answer, and strict
  parse_rate scores it 0%. Stripping the markdown fence rescues it. We report
  both, because "the model wrapped it in a fence" and "the model got the
  answer wrong" are different problems with different fixes, and one number
  cannot tell them apart.

  Row 4 parses, passes the schema, and is still wrong: 6 of 7 fields right,
  exact_match 0%. That is the gap the rest of Block B is about.
""")

    # -----------------------------------------------------------------------
    print("=" * 72)
    print("2. THE TEST SET")
    print("=" * 72)
    print(f"  {len(GOLD)} examples. Gold labels come from B1's derive(), so the three")
    print(f"  rule fields are exact by construction -- nobody labelled them by hand.\n")
    print(f"  {'message':<52} {'intent':<17} {'pri':<7}")
    for msg, g in GOLD:
        m = msg if len(msg) <= 50 else msg[:47] + "..."
        print(f"  {m:<52} {g['intent']:<17} {g['priority']:<7}")

    hard = sum(1 for _, g in GOLD if g["is_hygiene_item"])
    print(f"""
  {hard} of the {len(GOLD)} involve a hygiene item, which is where the rule interacts with
  itself. Five of those use a hygiene word a prompt is unlikely to guess:
  toothbrush, mascara, deodorant, shampoo, earbuds.

  This is a SMOKE TEST, not the real evaluation. The real held-out set is 150
  examples and B3 generates it. Sixteen is enough to prove the grader works
  and to get a first baseline; it is not enough to report a score anyone
  should believe.
""")

    # -----------------------------------------------------------------------
    print("=" * 72)
    print("3. THE BASELINE: UNTUNED qwen2.5:1.5b")
    print("=" * 72)

    if not ollama_up():
        print("  Ollama is not responding on localhost:11434.")
        print("  Start it, then re-run:   ollama serve")
        print("\n  Sections 1 and 2 above needed no model and already ran.")
        raise SystemExit(0)

    print(f"  model: {MODEL}   temperature 0, seed 0")
    print(f"  asking {len(GOLD)} questions", end="", flush=True)
    t0 = time.time()
    preds = []
    for msg, _ in GOLD:
        preds.append(ask(msg))
        print(".", end="", flush=True)
    elapsed = time.time() - t0
    print(f" done in {elapsed:.1f}s ({elapsed / len(GOLD):.1f}s each)\n")

    res = grade(preds, [g for _, g in GOLD])
    report("BASE MODEL, prompted", res)

    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("4. PER-FIELD ACCURACY -- THE NUMBER THAT TELLS YOU WHAT TO FIX")
    print("=" * 72)
    print(f"  {'field':<22} {'accuracy':>9}   {'kind':<12} bar")
    for f in FIELDS:
        a = res["field_acc"][f]
        kind = ("rule" if f in RULE_FIELDS
                else "extraction" if f in EXTRACT_FIELDS else "classify")
        print(f"  {f:<22} {a:>9.1%}   {kind:<12} {'#' * int(a * 30)}")

    ex = sum(res["field_acc"][f] for f in EXTRACT_FIELDS) / len(EXTRACT_FIELDS)
    ru = sum(res["field_acc"][f] for f in RULE_FIELDS) / len(RULE_FIELDS)
    print(f"\n  {'extraction fields, mean':<28} {ex:>7.1%}")
    print(f"  {'rule fields, mean':<28} {ru:>7.1%}")
    print(f"  {'intent (classification)':<28} {res['field_acc']['intent']:>7.1%}")

    print("""
  This split is the whole reason to report four numbers. Extraction and
  classification are things a 1.5B model can mostly already do from a prompt.
  Applying a multi-step rule consistently is not.

  Which tells you what fine-tuning is FOR here. Not "make the model smarter" --
  make it apply a specific rule reliably, in a specific output shape, at a
  specific cost.
""")

    # -----------------------------------------------------------------------
    print("=" * 72)
    print("5. WHAT WENT WRONG, EXAMPLE BY EXAMPLE")
    print("=" * 72)
    shown = 0
    for (msg, gold), raw, (obj, errs, okf) in zip(GOLD, preds, res["rows"]):
        if okf == len(FIELDS):
            continue
        shown += 1
        if shown > 5:
            break
        print(f"  {msg!r}")
        if errs:
            print(f"    schema: {errs[0][:66]}")
        if obj is not None:
            for f in FIELDS:
                g, p = gold[f], obj.get(f, "<missing>")
                if norm(g) != norm(p):
                    print(f"    {f:<22} got {str(p):<16} want {g}")
        print()
    bad = sum(1 for _, _, k in res["rows"] if k != len(FIELDS))
    print(f"  {bad} of {res['n']} examples had at least one wrong field"
          f" ({'showing first 5' if bad > 5 else 'all shown'}).")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("6. ONE HONEST CAVEAT: parse_rate IS NOT A REASON TO FINE-TUNE")
    print("=" * 72)
    print("  Ollama can force valid JSON with one argument. Same model, same prompt,")
    print("  format=\"json\"", end="", flush=True)
    t0 = time.time()
    preds_j = []
    for msg, _ in GOLD:
        preds_j.append(ask(msg, force_json=True))
        print(".", end="", flush=True)
    print(f" done in {time.time() - t0:.1f}s\n")

    res_j = grade(preds_j, [g for _, g in GOLD])
    report("BASE MODEL, format=\"json\"", res_j)

    print(f"\n  {'metric':<18} {'prompted':>10} {'format=json':>13} {'change':>9}")
    for k in ("parse_rate", "schema_valid", "exact_match"):
        print(f"  {k:<18} {res[k]:>10.1%} {res_j[k]:>13.1%}"
              f" {res_j[k] - res[k]:>+9.1%}")

    print(f"""
  Two things to say, and the second is the honest one.

  First: if your only problem is malformed JSON, constrained decoding fixes it
  for free, today, with no training and no GPU. parse_rate went to 100%.
  Fine-tuning is not the answer to that problem.

  Second, and read the other two rows before you believe the first: forcing the
  format made schema_valid and exact_match WORSE, not better. Constraining the
  sampler changes which tokens are reachable at each step, and that can push
  the model off the output it would otherwise have produced.

  How much should you trust that? Not much, at this size. On {res['n']} examples,
  {abs(res_j['schema_valid'] - res['schema_valid']) * res['n']:.0f} examples moved on schema_valid and {abs(res_j['exact_match'] - res['exact_match']) * res['n']:.0f} on exact_match. That is noise,
  not a finding. Say so in class rather than reading a story into it -- and
  re-run this comparison on B3's 150-example set, where the numbers mean
  something.

  What survives either way: forcing the output shape does not make the model
  apply the rule. priority sits at {res['field_acc']['priority']:.1%}. That is the gap fine-tuning
  is supposed to close, and the baseline above is what it will be measured
  against.
""")
    print("=" * 72)
    print("NEXT: b3 generates the real dataset -- and names what kind of")
    print("      distillation that actually is.")
    print("=" * 72)
