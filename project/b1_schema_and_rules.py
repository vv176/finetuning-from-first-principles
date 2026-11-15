"""
B1: The schema and the rules. Everything in Block B is measured against this file.

Block A used a one-field toy: {"intent": "damaged_item"}. The project needs a
real output, and the single most important decision in the whole module is that
THREE OF THE SEVEN FIELDS ARE DERIVED BY A PUBLISHED RULE, not by taste.

    intent                enum, 6 values      <- classification
    order_id              string | null       <- extraction
    item_name             string | null       <- extraction
    days_since_delivery   int | null          <- extraction
    is_hygiene_item       bool | null         <- RULE from item_name
    requested_action      enum | null         <- RULE from intent
    priority              enum, 4 values      <- RULE from intent + days

Why that matters, three ways:

  1. The grader needs no LLM judge. A rule is a Python function, so correctness
     is a comparison, not an opinion. B2 builds that grader on top of this file.

  2. You can prove the small model LEARNED THE RULE rather than memorised
     examples, by testing it on intent/item combinations that never appear in
     training. Section 5 builds exactly those cases.

  3. When a student disagrees with a label, there is a file to point at.

This file is the contract. It has no model, no network, no training. It runs in
well under a second and it is the first thing to run when anything downstream
looks wrong.

Run it:
    python3 b1_schema_and_rules.py
"""

import json
from typing import Any

from jsonschema import Draft202012Validator

# ---------------------------------------------------------------------------
# The enums. Closed sets, so "wrong value" is decidable.
# ---------------------------------------------------------------------------
INTENTS = [
    "damaged_item",
    "missing_delivery",
    "wrong_item",
    "late_delivery",
    "return_request",
    "general_question",
]

ACTIONS = ["replacement", "refund", "redeliver", "investigate", "answer"]

PRIORITIES = ["low", "medium", "high", "urgent"]

# Hygiene items cannot be resold, so they are never a replacement-from-stock
# decision and they change the action. The list is deliberately SHORT -- the
# test set contains hygiene words that are NOT in here, to check whether the
# model learned the idea or memorised the vocabulary.
HYGIENE_WORDS = [
    "toothbrush", "razor", "underwear", "socks", "lipstick", "mascara",
    "deodorant", "shampoo", "soap", "comb", "earphones", "earbuds",
]

RETURN_WINDOW_DAYS = 30


# ---------------------------------------------------------------------------
# The three rules. Each is a pure function with no model in sight.
# ---------------------------------------------------------------------------
def is_hygiene_item(item_name: str | None) -> bool | None:
    """RULE 1. Substring match against HYGIENE_WORDS, case-insensitive.

    Returns None when there is no item at all, because the field is then
    genuinely unknown rather than false. That distinction is graded.
    """
    if item_name is None:
        return None
    low = item_name.lower()
    return any(w in low for w in HYGIENE_WORDS)


def requested_action(intent: str) -> str | None:
    """RULE 2. One action per intent. A lookup, nothing more."""
    return {
        "damaged_item": "replacement",
        "missing_delivery": "investigate",
        "wrong_item": "replacement",
        "late_delivery": "investigate",
        "return_request": "refund",
        "general_question": "answer",
    }[intent]


def priority(intent: str, days_since_delivery: int | None,
             hygiene: bool | None = None) -> str:
    """RULE 3. Base priority from intent, then two adjustments.

    Base:
        damaged_item, wrong_item            high    (the customer has a bad item)
        missing_delivery                    high    (the customer has nothing)
        late_delivery                       medium
        return_request                      low
        general_question                    low

    Adjustment A -- staleness. days_since_delivery is None (unknown) leaves the
    base alone. Otherwise:
        <= 2 days   : bump up one   (fresh, fixable, the customer is angry now)
        > 30 days   : drop one      (outside the return window)

    Adjustment B -- hygiene bump. A hygiene item cannot be resold or replaced
    from stock, so it escalates one level.

    Order matters and is fixed: base, then staleness, then the hygiene bump.
    Both adjustments clamp at the ends, so 'urgent' cannot be exceeded and
    'low' cannot be undercut.
    """
    base = {
        "damaged_item": "high",
        "wrong_item": "high",
        "missing_delivery": "high",
        "late_delivery": "medium",
        "return_request": "low",
        "general_question": "low",
    }[intent]

    i = PRIORITIES.index(base)

    if days_since_delivery is not None:
        if days_since_delivery <= 2:
            i = min(len(PRIORITIES) - 1, i + 1)
        elif days_since_delivery > RETURN_WINDOW_DAYS:
            i = max(0, i - 1)

    if hygiene is True:
        i = min(len(PRIORITIES) - 1, i + 1)

    return PRIORITIES[i]


def derive(intent: str, order_id: str | None, item_name: str | None,
           days_since_delivery: int | None) -> dict[str, Any]:
    """Build the whole 7-field object from the 4 extracted fields.

    This is the labeller. B3 calls it after a frontier model has produced the
    extraction fields, which is what makes the dataset's derived fields exact.
    """
    hyg = is_hygiene_item(item_name)
    return {
        "intent": intent,
        "order_id": order_id,
        "item_name": item_name,
        "days_since_delivery": days_since_delivery,
        "is_hygiene_item": hyg,
        "requested_action": requested_action(intent),
        "priority": priority(intent, days_since_delivery, hyg),
    }


# ---------------------------------------------------------------------------
# The schema. Used by B2's grader for `schema_valid`.
# ---------------------------------------------------------------------------
SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent", "order_id", "item_name", "days_since_delivery",
        "is_hygiene_item", "requested_action", "priority",
    ],
    "properties": {
        "intent": {"enum": INTENTS},
        "order_id": {"type": ["string", "null"]},
        "item_name": {"type": ["string", "null"]},
        "days_since_delivery": {"type": ["integer", "null"], "minimum": 0},
        "is_hygiene_item": {"type": ["boolean", "null"]},
        "requested_action": {"enum": ACTIONS + [None]},
        "priority": {"enum": PRIORITIES},
    },
}

VALIDATOR = Draft202012Validator(SCHEMA)

FIELDS = SCHEMA["required"]
RULE_FIELDS = ["is_hygiene_item", "requested_action", "priority"]
EXTRACT_FIELDS = ["order_id", "item_name", "days_since_delivery"]


def schema_errors(obj: Any) -> list[str]:
    """Every reason obj is not a valid routing decision. Empty list = valid."""
    if not isinstance(obj, dict):
        return [f"top level is {type(obj).__name__}, not an object"]
    return [f"{'.'.join(str(p) for p in e.path) or '<root>'}: {e.message}"
            for e in sorted(VALIDATOR.iter_errors(obj), key=str)]


# ===========================================================================
# Everything below is the proof. It runs when you run the file.
# ===========================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("1. THE SCHEMA, AND WHAT IT REJECTS")
    print("=" * 72)

    good = derive("damaged_item", "8831", "water bottle", 1)
    print("  A valid object:\n")
    print("    " + json.dumps(good))
    print(f"\n  schema errors: {schema_errors(good)}\n")

    bad_cases = [
        ("an intent outside the enum",
         {**good, "intent": "broken_thing"}),
        ("days as a string",
         {**good, "days_since_delivery": "1"}),
        ("a missing field",
         {k: v for k, v in good.items() if k != "priority"}),
        ("an extra field",
         {**good, "confidence": 0.9}),
        ("negative days",
         {**good, "days_since_delivery": -3}),
        ("not an object at all",
         ["damaged_item"]),
    ]
    print(f"  {'what is wrong':<30} errors")
    for name, obj in bad_cases:
        errs = schema_errors(obj)
        print(f"  {name:<30} {len(errs)}  {errs[0][:46]}")

    print("""
  Every one of those is caught without a model and without an opinion. That is
  what B2's `schema_valid` metric counts.
""")

    # -----------------------------------------------------------------------
    print("=" * 72)
    print("2. RULE 2: requested_action IS A LOOKUP")
    print("=" * 72)
    print(f"  {'intent':<20} -> requested_action")
    for i in INTENTS:
        print(f"  {i:<20} -> {requested_action(i)}")
    print(f"\n  All {len(INTENTS)} intents covered. No default, no fallthrough --")
    print(f"  an unknown intent raises KeyError rather than guessing.")
    try:
        requested_action("not_a_real_intent")
    except KeyError as e:
        print(f"  requested_action('not_a_real_intent') -> KeyError({e})")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("3. RULE 1: is_hygiene_item, AND THE THREE-VALUED ANSWER")
    print("=" * 72)
    hyg_cases = ["water bottle", "Electric Toothbrush", "two pairs of SOCKS",
                 "wireless earbuds", "coffee mug", None]
    print(f"  {'item_name':<24} is_hygiene_item")
    for c in hyg_cases:
        print(f"  {str(c):<24} {is_hygiene_item(c)}")
    print("""
  Three possible answers, not two. None means 'there is no item named', which
  is different from False ('there is an item and it is not hygiene'). The
  grader treats them as different values, so a model that guesses False when it
  should say null loses that field.
""")

    # -----------------------------------------------------------------------
    print("=" * 72)
    print("4. RULE 3: priority, EVERY COMBINATION")
    print("=" * 72)
    day_cols = [None, 1, 2, 5, 20, 31, 90]
    print(f"  {'intent':<18} {'hyg':<6}" + "".join(f"{str(d):>8}" for d in day_cols))
    print(f"  {'-' * 18} {'-' * 6}" + "".join(f"{'-' * 8}" for _ in day_cols))
    for i in INTENTS:
        for h in (False, True):
            row = "".join(f"{priority(i, d, h):>8}" for d in day_cols)
            print(f"  {i:<18} {str(h):<6}{row}")

    print("""
  Read the columns. 'None' is the base priority, untouched. Days 1 and 2 are
  bumped up. Day 31 and beyond are dropped. Each hygiene row sits one level
  above the row above it, except where it has hit the ceiling.

  Read the rows too: return_request at day 1 is 'medium', and at day 31 it is
  'low'. That is a 30-day return window expressed as a priority, which is the
  kind of thing a prompt gets wrong and a rule cannot.

  Notice the hygiene rows are NOT constant. That is deliberate. A floor at
  'high' would have made them constant, and a model could then score well on
  this field by learning one word instead of the rule.
""")

    # -----------------------------------------------------------------------
    print("=" * 72)
    print("5. THE COMBINATIONS A MODEL CANNOT MEMORISE")
    print("=" * 72)
    print("""  These exist so the held-out test set can prove the model learned the
  RULE. Each one is a cell of the table above that B3's generator is told
  never to produce in training.
""")
    unseen = [
        ("general_question", None, "toothbrush", 1,
         "a hygiene item on the lowest-priority intent, bumped twice"),
        ("return_request", "9001", "razor", 45,
         "hygiene bump fights the out-of-window drop"),
        ("late_delivery", None, None, 2,
         "no item, no order id, fresh"),
        ("missing_delivery", "7742", "mascara", 90,
         "hygiene + very stale + high base"),
    ]
    for intent, oid, item, days, why in unseen:
        obj = derive(intent, oid, item, days)
        print(f"  {why}")
        print(f"    {json.dumps(obj)}")
        print(f"    -> priority {obj['priority']!r}, "
              f"hygiene {obj['is_hygiene_item']}, "
              f"action {obj['requested_action']!r}\n")

    print("""  Look at the second one. return_request has base 'low'. 45 days tries to
  drop it, but it is already at the bottom so the clamp holds it. Then the
  hygiene bump lifts it one level. Two adjustments pulling opposite ways, and
  the published order of operations plus the clamp decide the answer.

  No amount of pattern-matching on training examples gets that right. Either
  the model learned the rule or it did not, and this case tells you which.
""")

    # -----------------------------------------------------------------------
    print("=" * 72)
    print("6. THE ASSERTIONS")
    print("=" * 72)

    checks = 0

    def check(got, want, label):
        global checks
        assert got == want, f"{label}: got {got!r}, want {want!r}"
        checks += 1

    # rule 2, all six
    for i, a in [("damaged_item", "replacement"), ("missing_delivery", "investigate"),
                 ("wrong_item", "replacement"), ("late_delivery", "investigate"),
                 ("return_request", "refund"), ("general_question", "answer")]:
        check(requested_action(i), a, f"action({i})")

    # rule 1
    check(is_hygiene_item(None), None, "hygiene(None)")
    check(is_hygiene_item("water bottle"), False, "hygiene(bottle)")
    check(is_hygiene_item("Toothbrush"), True, "hygiene(case)")
    check(is_hygiene_item("2 x razor blades"), True, "hygiene(substring)")

    # rule 3 base
    check(priority("damaged_item", None), "high", "base damaged")
    check(priority("late_delivery", None), "medium", "base late")
    check(priority("return_request", None), "low", "base return")
    check(priority("general_question", None), "low", "base question")

    # rule 3 staleness
    check(priority("damaged_item", 1), "urgent", "fresh bumps")
    check(priority("damaged_item", 2), "urgent", "day 2 bumps")
    check(priority("damaged_item", 3), "high", "day 3 does not")
    check(priority("damaged_item", 31), "medium", "stale drops")
    check(priority("return_request", 90), "low", "cannot drop below low")
    check(priority("missing_delivery", 1), "urgent", "fresh missing")

    # rule 3 hygiene bump
    check(priority("general_question", None, True), "medium", "hygiene bumps low")
    check(priority("general_question", 1, True), "high", "two bumps stack")
    check(priority("return_request", 45, True), "medium", "clamp then bump")
    check(priority("late_delivery", None, True), "high", "medium bumps to high")
    check(priority("damaged_item", 1, True), "urgent", "bump clamps at urgent")
    check(priority("general_question", None, False), "low", "False does not bump")
    check(priority("general_question", None, None), "low", "None does not bump")

    # clamping at both ends
    check(priority("damaged_item", 1), PRIORITIES[-1], "cannot exceed urgent")
    check(priority("general_question", 90), PRIORITIES[0], "cannot go below low")

    # derive() agrees with the parts
    for intent in INTENTS:
        for item in (None, "water bottle", "toothbrush"):
            for days in (None, 1, 10, 60):
                o = derive(intent, "1", item, days)
                check(schema_errors(o), [], f"derive valid {intent}/{item}/{days}")
                check(o["requested_action"], requested_action(intent), "derive action")
                check(o["is_hygiene_item"], is_hygiene_item(item), "derive hygiene")
                check(o["priority"],
                      priority(intent, days, is_hygiene_item(item)), "derive priority")

    # the rule is a pure function: same inputs, same output, always
    check(priority("damaged_item", 1, True), priority("damaged_item", 1, True),
          "deterministic")

    print(f"  {checks} assertions pass.\n")
    print("  Nothing here needed a model, a GPU, a network call or an API key.")
    print("  This file is the ground truth the rest of Block B is scored against.")
    print("\n" + "=" * 72)
    print("NEXT: b2 turns this into a grader and measures the BASE model with it.")
    print("=" * 72)
