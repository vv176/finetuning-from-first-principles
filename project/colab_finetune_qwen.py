"""
The real fine-tune: Qwen2.5-1.5B-Instruct on a Colab T4.

One file, five parts. Run it top to bottom in Colab, or paste one part per cell.

    PART 1  the model          load it, check the GPU, verify the chat template
    PART 2  the data           schema + rules + generate train/test jsonl
    PART 3  the grader         four deterministic metrics, and the BEFORE score
    PART 4  the fine-tune      LoRA, 0.4% trainable, A3's four lines
    PART 5  the verdict        AFTER score, before/after table, save the adapter

Block A built every mechanism here by hand on GPT-2. This file uses the real
libraries on a real model, and the only new thing is scale.

Where each part came from:

    A2  mask the prompt, grade the answer only
    A3  loss -> backward -> step -> zero_grad
    A4  LoRA: W frozen, A and B trained
    A6  apply_chat_template, and the exact index where the answer starts
    A7  a held-out test set must be held out along PHRASING, not ids
    B1  the schema and the three derived fields
    B2  the grader, and the baseline this file is measured against

Hardware: a free Colab T4 (16 GB). Runtime/Change runtime type/T4 GPU.
Total runtime: about 25 minutes, most of it PART 2 and PART 4.
"""

# ===========================================================================
# PART 1 OF 5 -- THE MODEL
# ===========================================================================
# What this part does:
#   1. installs the libraries Colab does not already have
#   2. confirms a GPU is attached, and picks fp16 or bf16 correctly for it
#   3. loads Qwen2.5-1.5B-Instruct and reports what it costs in VRAM
#   4. verifies the chat template, including A6's prefix check
#   5. asks the base model one task question, so you see the starting point
#
# Set DATA_ONLY=1 to skip all of it and run PART 2 alone, on a laptop, with no
# GPU and no model. That is the recommended way to generate the data: your API
# key stays on your machine and you upload only the two .jsonl files to Colab.
#
#     DATA_ONLY=1 python3 colab_finetune_qwen.py
# ---------------------------------------------------------------------------

# In Colab, uncomment this line and run it once. It takes about 40 seconds.
# !pip install -q -U "transformers==4.57.1" peft accelerate jsonschema openai

import json
import os
import time

DATA_ONLY = os.environ.get("DATA_ONLY") == "1"

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = ("You are a support router. Read the customer message and reply with "
          "one JSON object. Reply with JSON only.")

tokenizer = model = None
DTYPE = BF16 = vram = None


def ask(message, system=SYSTEM, max_new_tokens=120):
    """One greedy generation. Greedy, so the same input always gives the same output."""
    ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": message}],
        add_generation_prompt=True, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


if DATA_ONLY:
    print("=" * 72)
    print("DATA_ONLY=1  --  PART 1 skipped. No GPU, no model, no training.")
    print("  This run only generates train.jsonl and test.jsonl.")
    print("=" * 72)
else:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("=" * 72)
    print("PART 1.1  THE GPU")
    print("=" * 72)

    if not torch.cuda.is_available():
        raise SystemExit(
            "  No GPU attached.\n"
            "  Colab: Runtime -> Change runtime type -> T4 GPU -> Save, then rerun.\n"
            "  On a laptop, generate the data instead:  DATA_ONLY=1 python3 "
            + os.path.basename(__file__))

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    # bf16 needs compute capability 8.0 (Ampere). A T4 is 7.5, so a T4 gets fp16.
    # Asking for bf16 on a T4 does not warn -- it silently crawls, or fails later.
    BF16 = major >= 8
    DTYPE = torch.bfloat16 if BF16 else torch.float16

    print(f"  device              : {name}")
    print(f"  compute capability  : {major}.{minor}")
    print(f"  VRAM                : {vram:.1f} GB")
    print(f"  dtype for this run  : {'bfloat16' if BF16 else 'float16'}"
          f"   ({'Ampere or newer' if BF16 else 'Turing: no bf16, so fp16'})")

    print(f"""
  Why this matters, with A3's arithmetic for a 1.5B model:

    full fine-tune, 16-bit     ~24 GB     does NOT fit {vram:.0f} GB
    LoRA, 16-bit base           ~6 GB     fits, with room to spare
    QLoRA, 4-bit base           ~0.8 GB   not needed at this size

  So this file uses plain LoRA on a 16-bit base. 4-bit becomes necessary
  around 7B, not here.
""")

    print("=" * 72)
    print("PART 1.2  LOADING THE MODEL")
    print("=" * 72)
    print("  Downloading about 3.1 GB on the first run, then cached.\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map="cuda")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters          : {n_params:,}")
    print(f"  bytes per parameter : 2   (16-bit)")
    print(f"  weights in VRAM     : {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"  VRAM still free     : {(vram - torch.cuda.memory_allocated()) / 1e9:.2f} GB")
    print("""
  No API key was needed for this. Qwen2.5 is Apache-2.0 and ungated, so
  from_pretrained just downloads it. A Hugging Face token is needed only for
  gated models (Llama, Gemma) or to upload your own adapter.
""")

    print("=" * 72)
    print("PART 1.3  THE CHAT TEMPLATE  (A6, on the real model)")
    print("=" * 72)

    EXAMPLE_MSG = "Order ORD-4821 - my water bottle arrived cracked, I want a refund."
    EXAMPLE_ANSWER = '{"order_id": "ORD-4821", "intent": "damaged_item"}'

    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": EXAMPLE_MSG}]

    gen_prompt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                               add_generation_prompt=True)
    gen_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        msgs + [{"role": "assistant", "content": EXAMPLE_ANSWER}], tokenize=False)
    full_ids = tokenizer(full_text).input_ids

    print("  what the model actually reads:\n")
    print("  " + gen_prompt.replace("\n", "\\n\n  "))

    print(f"\n  generation prompt   : {len(gen_ids)} tokens")
    print(f"  prompt + answer     : {len(full_ids)} tokens")
    print(f"  graded tokens       : {len(full_ids) - len(gen_ids)}")
    print(f"  is the prompt an exact PREFIX of the full sequence? "
          f"{full_ids[:len(gen_ids)] == gen_ids}")

    print(f"""
  That True is the load-bearing fact of PART 4.

  Because the generation prompt is an exact prefix, PART 4 can mask the prompt
  by LENGTH alone:  labels[:{len(gen_ids)}] = -100. If it were not a prefix, masking by
  length would silently grade the wrong tokens, and nothing would warn you.

  Two special tokens to know:
    <|im_start|> = {tokenizer.convert_tokens_to_ids('<|im_start|>')}      marks the start of a turn
    <|im_end|>   = {tokenizer.convert_tokens_to_ids('<|im_end|>')}      marks the end of a turn

  <|im_end|> is one of the graded tokens. That is how the model learns to STOP.
  Leave it out of the labels and your fine-tune never ends its turn.
""")

    print("=" * 72)
    print("PART 1.4  THE BASE MODEL, BEFORE ANY TRAINING")
    print("=" * 72)
    print(f"  message : {EXAMPLE_MSG}\n")
    print(f"  reply   : {ask(EXAMPLE_MSG)!r}")

    print("""
  Read that reply carefully. It is a 1.5B instruct model, so it probably did
  emit JSON -- Qwen's post-training already taught it to follow instructions.

  That is the honest starting point, and it is far above GPT-2's floor of 0% in
  A7. The question PART 3 answers is not "can it emit JSON" but "does it get
  the FIELDS right", and especially the three fields decided by a rule.

  B2 already measured that on this exact model, with no fine-tuning:

      parse_rate     37.5%        (100% after stripping markdown fences)
      schema_valid   87.5%
      priority       37.5%        <- the rule field
      exact_match    18.8%

  PART 3 reproduces those numbers here, so PART 5 has something to compare to.
""")

    print("=" * 72)
    print("END OF PART 1.  Next: PART 2 builds the schema, the rules, and the data.")
    print("=" * 72)


# ===========================================================================
# PART 2 OF 5 -- THE DATA
# ===========================================================================
# What this part does:
#   1. states the 7-field schema and the three rules  (B1, inlined)
#   2. builds a spec grid: intent, item, days, order id -- chosen, not guessed
#   3. asks a frontier model to write ONLY the message text for each spec
#   4. computes all 7 labels with derive(), so no label is ever a model's guess
#   5. holds the test set out along PHRASING, which is A7's lesson
#   6. writes train.jsonl and test.jsonl
#
# The one idea that matters here: the model writes the QUESTION, the rule
# writes the ANSWER. Three of the seven fields follow a published function, so
# the dataset's labels are exact and the grader needs no LLM judge.
# ---------------------------------------------------------------------------

import os
import random
import re
from jsonschema import Draft202012Validator

print("=" * 72)
print("PART 2.1  THE SCHEMA AND THE RULES")
print("=" * 72)

INTENTS = ["damaged_item", "missing_delivery", "wrong_item",
           "late_delivery", "return_request", "general_question"]
ACTIONS = ["replacement", "refund", "redeliver", "investigate", "answer"]
PRIORITIES = ["low", "medium", "high", "urgent"]

# Deliberately SHORT. The rule IS this list -- see the note under 2.5.
HYGIENE_WORDS = ["toothbrush", "razor", "underwear", "socks", "lipstick",
                 "mascara", "deodorant", "shampoo", "soap", "comb",
                 "earphones", "earbuds"]
RETURN_WINDOW_DAYS = 30


def is_hygiene_item(item_name):
    """RULE 1. Substring match against a fixed list."""
    if item_name is None:
        return None
    low = item_name.lower()
    return any(w in low for w in HYGIENE_WORDS)


def requested_action(intent):
    """RULE 2. One action per intent. A lookup, nothing more."""
    return {
        "damaged_item": "replacement",
        "missing_delivery": "investigate",
        "wrong_item": "replacement",
        "late_delivery": "investigate",
        "return_request": "refund",
        "general_question": "answer",
    }[intent]


def priority(intent, days_since_delivery, hygiene=None):
    """RULE 3. Base from intent, then two adjustments, in a fixed order.

    base:        damaged/wrong/missing = high, late = medium, return/question = low
    staleness:   <= 2 days  -> up one       > 30 days -> down one      None -> no change
    hygiene:     True -> up one
    Both adjustments clamp at 'low' and 'urgent'.
    """
    base = {
        "damaged_item": "high", "wrong_item": "high", "missing_delivery": "high",
        "late_delivery": "medium", "return_request": "low", "general_question": "low",
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


def derive(intent, order_id, item_name, days_since_delivery):
    """The labeller. Four extracted fields in, all seven out."""
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


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "order_id", "item_name", "days_since_delivery",
                 "is_hygiene_item", "requested_action", "priority"],
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

print("  7 fields. Four are read off the message, three follow a rule.\n")
print(f"  {'field':<22} {'source':<14} note")
print(f"  {'-' * 22} {'-' * 14} {'-' * 28}")
for f, src, note in [
    ("intent", "the model", "classification, 6 values"),
    ("order_id", "the model", "extraction, copied"),
    ("item_name", "the model", "extraction, copied"),
    ("days_since_delivery", "the model", "extraction, int or null"),
    ("is_hygiene_item", "RULE 1", "from item_name"),
    ("requested_action", "RULE 2", "from intent"),
    ("priority", "RULE 3", "from intent + days + hygiene"),
]:
    print(f"  {f:<22} {src:<14} {note}")

print("\n  RULE 3 has texture, which is what makes it worth learning:\n")
print(f"  {'intent':<18} {'days':>5} {'hygiene':>8}   {'base':<8} {'priority':<8} what moved it")
for intent, days, item in [
    ("damaged_item", 1, "bottle"), ("damaged_item", 10, "bottle"),
    ("damaged_item", 40, "bottle"), ("damaged_item", 10, "razor"),
    ("return_request", 1, "razor"), ("late_delivery", None, "lamp"),
    ("general_question", 40, "mug"),
]:
    h = is_hygiene_item(item)
    base = priority(intent, None, False)
    moves = []
    if days is not None and days <= 2:
        moves.append("fresh: up one")
    elif days is not None and days > RETURN_WINDOW_DAYS:
        moves.append("stale: down one")
    if h:
        moves.append("hygiene: up one")
    print(f"  {intent:<18} {str(days):>5} {str(h):>8}   {base:<8} "
          f"{priority(intent, days, h):<8} {', '.join(moves) or 'nothing'}")

print("""
  Look at rows 1 to 4. Same intent, four different priorities. A model cannot
  score well on this field by learning one word -- it has to apply the rule.
""")

print("=" * 72)
print("PART 2.2  THE SPEC GRID")
print("=" * 72)

N_TRAIN, N_TEST = 240, 60
BATCH = 12                      # specs per API call

# Items are held out too, so extraction is tested on words never trained on.
# Each pool mixes hygiene and non-hygiene items.
TRAIN_ITEMS = ["water bottle", "desk lamp", "coffee mug", "bluetooth speaker",
               "phone charger", "blender", "electric kettle", "monitor stand",
               "toothbrush", "razor", "wool socks", "shampoo bottle"]
TEST_ITEMS = ["travel kettle", "laptop sleeve", "ceramic plate", "wireless mouse",
              "air fryer", "bath towel",
              # ON the HYGIENE_WORDS list -> correct label True
              "deodorant stick", "lipstick", "cotton underwear", "earbuds",
              # hygiene products NOT on the list -> correct label False.
              # This is the trap described at the end of PART 2.
              "toothpaste", "face wash", "shaving foam", "cotton buds"]

# Phrasing is held out, which is A7's lesson. Train and test ask for different
# registers, so a model that memorised sentence patterns will be caught.
TRAIN_STYLES = [
    "polite and complete sentences",
    "short and businesslike",
    "mildly annoyed but civil",
    "chatty, with one aside",
]
TEST_STYLES = [
    "terse, lowercase, no punctuation at the end, like a phone message",
    "formal and slightly stiff, as if copied from a complaint form",
]

# days_since_delivery options, chosen to exercise every branch of RULE 3.
DAY_CHOICES = [None, 0, 1, 2, 5, 9, 14, 27, 31, 45, 60]


def make_specs(n, items, rng):
    """Pick intent, item, days and an order id for each example. No model yet."""
    specs = []
    for k in range(n):
        intent = INTENTS[k % len(INTENTS)]                  # balanced classes
        item = rng.choice(items)
        days = rng.choice(DAY_CHOICES)
        if intent == "general_question" and rng.random() < 0.5:
            days = None                                     # often no delivery yet
        specs.append({
            "intent": intent,
            "item_name": item,
            "days_since_delivery": days,
            "order_id": "ORD-" + str(rng.randint(10000, 99999)),
        })
    return specs


rng_train = random.Random(0)
rng_test = random.Random(99)
train_specs = make_specs(N_TRAIN, TRAIN_ITEMS, rng_train)
test_specs = make_specs(N_TEST, TEST_ITEMS, rng_test)

print(f"  train specs : {len(train_specs)}   items from a pool of {len(TRAIN_ITEMS)}")
print(f"  test specs  : {len(test_specs)}   items from a DISJOINT pool of {len(TEST_ITEMS)}")
print(f"  no item, and no phrasing style, appears in both\n")
print("  three train specs:")
for s in train_specs[:3]:
    print(f"    {s}")

print("""
  Nothing has asked a model anything yet. The spec is CHOSEN, which is what
  makes the labels exact: we are not inferring days_since_delivery from text,
  we picked it and then asked for text that contains it.
""")


print("=" * 72)
print("PART 2.3  WRITING THE MESSAGES")
print("=" * 72)

WRITER_SYSTEM = """You write realistic customer support messages for an online retailer.

You will be given numbered specifications. For each one, write ONE customer message.

Hard rules for every message:
- It must contain the order id EXACTLY as given.
- It must name the item EXACTLY as given.
- If a day count is given, the message must state that number so a reader can
  recover it (for example "delivered 5 days ago", "it has been 5 days").
- If no day count is given, do not mention any time period at all.
- The message must clearly be about the given problem, but must NOT use the
  problem's label as a word.
- One to three sentences. Vary length and punctuation across the batch.
- Plain text. No JSON, no surrounding quotes, no preamble, no explanation.

Output exactly one line per specification, formatted as:
<number>. <message>
Nothing else."""

PROBLEM_WORDING = {
    "damaged_item": "the item arrived physically damaged",
    "missing_delivery": "the item was marked delivered but never arrived",
    "wrong_item": "a different item arrived than the one ordered",
    "late_delivery": "the delivery is overdue and has not arrived yet",
    "return_request": "the customer wants to send the item back for money",
    "general_question": "the customer is asking a question, not reporting a fault",
}


def spec_lines(specs, start=1):
    out = []
    for i, s in enumerate(specs, start=start):
        days = ("no day count" if s["days_since_delivery"] is None
                else f"{s['days_since_delivery']} days")
        out.append(f"{i}. order id {s['order_id']}, item \"{s['item_name']}\", "
                   f"{days}, problem: {PROBLEM_WORDING[s['intent']]}")
    return "\n".join(out)


def load_key(name):
    """Environment variable first, then Colab's Secrets panel (the key icon)."""
    v = os.environ.get(name)
    if v:
        return v
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v:
            os.environ[name] = v
            return v
    except Exception:
        pass
    return None


# Either provider works -- this step only needs something that writes English.
PROVIDER = ("anthropic" if load_key("ANTHROPIC_API_KEY")
            else "openai" if load_key("OPENAI_API_KEY")
            else None)
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def write_messages_with_api(specs, styles, label):
    """One API call per BATCH specs. Returns a list the same length as specs,
    with None where generation failed or the message broke a hard rule."""
    import anthropic
    client = anthropic.Anthropic()
    out = [None] * len(specs)
    in_tok = out_tok = 0

    for start in range(0, len(specs), BATCH):
        chunk = specs[start:start + BATCH]
        style = styles[(start // BATCH) % len(styles)]
        prompt = (f"Write these {len(chunk)} messages in this register: {style}.\n\n"
                  + spec_lines(chunk, start=1))
        resp = client.messages.create(
            model="claude-opus-5-5",
            max_tokens=2000,
            output_config={"effort": "low"},        # plain writing, no deep reasoning
            system=WRITER_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        in_tok += resp.usage.input_tokens
        out_tok += resp.usage.output_tokens
        text = "".join(b.text for b in resp.content if b.type == "text")

        for line in text.splitlines():
            m = re.match(r"\s*(\d+)\.\s*(.+?)\s*$", line)
            if not m:
                continue
            idx = start + int(m.group(1)) - 1
            if 0 <= idx < len(specs):
                out[idx] = m.group(2)
        print(f"  {label}: {sum(x is not None for x in out)}/{len(specs)} written", end="\r")

    print(f"  {label}: {sum(x is not None for x in out)}/{len(specs)} written      ")
    cost = in_tok / 1e6 * 4.00 + out_tok / 1e6 * 20.00     # claude-opus-5-5 rates
    print(f"  tokens: {in_tok:,} in, {out_tok:,} out   approx ${cost:.2f}")
    return out


def write_messages_with_openai(specs, styles, label):
    """Same contract as the Anthropic writer, via the OpenAI SDK."""
    from openai import OpenAI
    client = OpenAI()
    out = [None] * len(specs)
    in_tok = out_tok = 0
    cap_field = "max_tokens"            # newer OpenAI models want max_completion_tokens

    for start in range(0, len(specs), BATCH):
        chunk = specs[start:start + BATCH]
        style = styles[(start // BATCH) % len(styles)]
        prompt = (f"Write these {len(chunk)} messages in this register: {style}.\n\n"
                  + spec_lines(chunk, start=1))
        kw = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "system", "content": WRITER_SYSTEM},
                         {"role": "user", "content": prompt}],
            cap_field: 2000,
        }
        try:
            resp = client.chat.completions.create(**kw)
        except Exception as e:
            if cap_field == "max_tokens" and "max_completion_tokens" in str(e):
                cap_field = "max_completion_tokens"      # retry once, then keep it
                kw.pop("max_tokens")
                kw["max_completion_tokens"] = 2000
                resp = client.chat.completions.create(**kw)
            else:
                raise
        in_tok += resp.usage.prompt_tokens
        out_tok += resp.usage.completion_tokens
        text = resp.choices[0].message.content or ""

        for line in text.splitlines():
            m = re.match(r"\s*(\d+)\.\s*(.+?)\s*$", line)
            if not m:
                continue
            idx = start + int(m.group(1)) - 1
            if 0 <= idx < len(specs):
                out[idx] = m.group(2)
        print(f"  {label}: {sum(x is not None for x in out)}/{len(specs)} written", end="\r")

    print(f"  {label}: {sum(x is not None for x in out)}/{len(specs)} written      ")
    print(f"  tokens: {in_tok:,} in, {out_tok:,} out   "
          f"(model {OPENAI_MODEL}; check your provider's pricing)")
    return out


# --- fallback: templates. Disjoint between train and test, but see the warning.
TRAIN_TEMPLATES = {
    "damaged_item": ["my {item} arrived cracked", "the {item} was broken in the box"],
    "missing_delivery": ["my {item} never showed up", "no sign of the {item}"],
    "wrong_item": ["you sent the wrong {item}", "this is not the {item} I ordered"],
    "late_delivery": ["the {item} is overdue", "still waiting on the {item}"],
    "return_request": ["I want to return the {item}", "please refund the {item}"],
    "general_question": ["is the {item} dishwasher safe", "what warranty does the {item} have"],
}
TEST_TEMPLATES = {
    "damaged_item": ["the {item} turned up in pieces"],
    "missing_delivery": ["tracking says the {item} arrived but it is not here"],
    "wrong_item": ["I received something else instead of the {item}"],
    "late_delivery": ["where has the {item} got to"],
    "return_request": ["can I send the {item} back for my money"],
    "general_question": ["how do I register the {item}"],
}


def write_messages_with_templates(specs, templates, rng):
    out = []
    for s in specs:
        body = rng.choice(templates[s["intent"]]).format(item=s["item_name"])
        when = ("" if s["days_since_delivery"] is None
                else f", it has been {s['days_since_delivery']} days")
        out.append(f"Order {s['order_id']}: {body}{when}.")
    return out


print("=" * 72)
print("PART 2.4  CHECKING WHAT CAME BACK")
print("=" * 72)
print("  A generated message is only usable if the facts we chose are IN it.\n")


def keep(spec, msg):
    """Reject anything a reader could not extract the chosen fields from."""
    if not msg:
        return "no message"
    if spec["order_id"] not in msg:
        return "order id missing"
    if spec["item_name"].lower() not in msg.lower():
        return "item name missing"
    d = spec["days_since_delivery"]
    if d is not None and str(d) not in msg:
        return "day count missing"
    return None


def build(specs, msgs, label):
    rows, dropped = [], {}
    for s, m in zip(specs, msgs):
        why = keep(s, m)
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        gold = derive(s["intent"], s["order_id"], s["item_name"],
                      s["days_since_delivery"])
        errs = sorted(VALIDATOR.iter_errors(gold), key=str)
        if errs:
            dropped["schema invalid"] = dropped.get("schema invalid", 0) + 1
            continue
        rows.append({"message": m, "gold": gold})
    print(f"  {label:<6}: kept {len(rows)} of {len(specs)}"
          + (f"   dropped {dict(dropped)}" if dropped else ""))
    return rows


REUSE = (os.path.exists("train.jsonl") and os.path.exists("test.jsonl")
         and os.environ.get("REGENERATE") != "1")

if REUSE:
    train = [json.loads(l) for l in open("train.jsonl") if l.strip()]
    test = [json.loads(l) for l in open("test.jsonl") if l.strip()]
    print(f"  train.jsonl and test.jsonl already exist -- reusing them.")
    print(f"  train {len(train)}, test {len(test)}.  "
          f"Set REGENERATE=1 to build them again.\n")
else:
    if PROVIDER == "anthropic":
        print("  ANTHROPIC_API_KEY found. Claude writes the messages.\n")
        train_msgs = write_messages_with_api(train_specs, TRAIN_STYLES, "train")
        test_msgs = write_messages_with_api(test_specs, TEST_STYLES, "test ")
    elif PROVIDER == "openai":
        print(f"  OPENAI_API_KEY found. {OPENAI_MODEL} writes the messages.")
        print("  Override the model with OPENAI_MODEL=... if you prefer another.\n")
        train_msgs = write_messages_with_openai(train_specs, TRAIN_STYLES, "train")
        test_msgs = write_messages_with_openai(test_specs, TEST_STYLES, "test ")
    else:
        print("""  NO API KEY FOUND, so falling back to templates.

  Read this before trusting the result. A7 measured exactly what templates do:
  the model scored 100% on a held-out test set and 33% on phrasings it had not
  seen, because it learned 24 sentence patterns instead of the task. The
  fallback below uses disjoint templates for train and test, which helps, but
  it is still a handful of patterns.

  To do this properly, set ANTHROPIC_API_KEY or OPENAI_API_KEY and rerun. The
  recommended place to do that is your own machine, not Colab:

      DATA_ONLY=1 OPENAI_API_KEY=sk-... python3 colab_finetune_qwen.py

  then upload train.jsonl and test.jsonl to Colab. Your key never leaves your
  laptop that way.
""")
        train_msgs = write_messages_with_templates(train_specs, TRAIN_TEMPLATES, rng_train)
        test_msgs = write_messages_with_templates(test_specs, TEST_TEMPLATES, rng_test)

    train = build(train_specs, train_msgs, "train")
    test = build(test_specs, test_msgs, "test")

print(f"""
  Every kept label was produced by derive(), not by a model. The three rule
  fields are therefore exact by construction, and the grader in PART 3 can
  compare them instead of judging them.
""")


print("=" * 72)
print("PART 2.5  THE FILES")
print("=" * 72)

if not REUSE:
    with open("train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open("test.jsonl", "w") as f:
        for r in test:
            f.write(json.dumps(r) + "\n")

print(f"  train.jsonl : {len(train)} examples, {os.path.getsize('train.jsonl'):,} bytes")
print(f"  test.jsonl  : {len(test)} examples, {os.path.getsize('test.jsonl'):,} bytes\n")
print("  two examples, as the model will see them:\n")
for r in train[:2]:
    print(f"    {r['message']}")
    print(f"      -> {json.dumps(r['gold'])}\n")

hyg_train = sum(r["gold"]["is_hygiene_item"] for r in train)
print(f"  {'priority distribution, train':<34}: "
      f"{ {p: sum(r['gold']['priority'] == p for r in train) for p in PRIORITIES} }")
print(f"  {'hygiene items, train':<34}: {hyg_train} of {len(train)}")

on_list = sorted({r["gold"]["item_name"] for r in test if r["gold"]["is_hygiene_item"]})
off_list = sorted({r["gold"]["item_name"] for r in test
                   if not r["gold"]["is_hygiene_item"]
                   and any(w in r["gold"]["item_name"].lower()
                           for w in ["tooth", "wash", "shav", "cotton bud", "towel"])})
print(f"  {'test items scored hygiene=True':<34}: {on_list}")
print(f"  {'hygiene-LOOKING but scored False':<34}: {off_list}")

print(f"""
  One trap worth naming before PART 3 reports a number.

  RULE 1 is a fixed list of {len(HYGIENE_WORDS)} words. The two rows printed just above are
  the trap: toothpaste, face wash, shaving foam and cotton buds are obviously
  hygiene products, and their correct label is False, because none of those
  words is on the list. A model that generalises the IDEA will answer True and
  be marked wrong -- the rule, not the idea, is the published contract.

  That is deliberate. It is the difference between learning a policy and
  guessing at one, and it is the kind of thing a real routing system is
  judged on.
""")

print("=" * 72)
print("END OF PART 2.  Next: PART 3 builds the grader and scores the base model.")
print("=" * 72)

if DATA_ONLY:
    print("""
  DATA_ONLY=1, so stopping here.

  Next: upload train.jsonl and test.jsonl to Colab (drag them into the file
  panel on the left), put this script beside them, and run it there WITHOUT
  DATA_ONLY. PART 2 will see the two files and skip generation, so no API key
  is ever needed on Colab.
""")
    raise SystemExit(0)


# ===========================================================================
# PART 3 OF 5 -- THE GRADER, AND THE BEFORE SCORE
# ===========================================================================
# What this part does:
#   1. writes the one prompt used for EVERY measurement in this file
#   2. parses the model's reply into JSON, counting repairs separately
#   3. scores four things, all by comparison -- no LLM judge anywhere
#   4. runs it on the base model, which is the number PART 5 is measured against
#
# The grader has to exist before the fine-tune, not after. Otherwise you tune
# against a number you invented once you had seen the output.
# ---------------------------------------------------------------------------

print("=" * 72)
print("PART 3.1  THE PROMPT")
print("=" * 72)

ROUTER_SYSTEM = f"""You are a support router. Read the customer message and reply with exactly one JSON object and nothing else.

Fields, all seven required:
  intent               one of: {", ".join(INTENTS)}
  order_id             the order id as written in the message, or null
  item_name            the item as written in the message, or null
  days_since_delivery  integer number of days stated in the message, or null
  is_hygiene_item      see rule 1
  requested_action     see rule 2
  priority             see rule 3

Rule 1 - is_hygiene_item is true only if the item name contains one of these exact words: {", ".join(HYGIENE_WORDS)}. Otherwise false. Null only if item_name is null.

Rule 2 - requested_action is fixed per intent: damaged_item and wrong_item give replacement, missing_delivery and late_delivery give investigate, return_request gives refund, general_question gives answer.

Rule 3 - priority starts from the intent: damaged_item, wrong_item and missing_delivery start at high, late_delivery starts at medium, return_request and general_question start at low. Then apply, in this order, on the scale low < medium < high < urgent:
  if days_since_delivery is 2 or less, move up one level
  else if days_since_delivery is more than {RETURN_WINDOW_DAYS}, move down one level
  if is_hygiene_item is true, move up one level
Never go above urgent or below low.

Reply with the JSON object only. No markdown fences, no explanation."""

print(f"  {len(ROUTER_SYSTEM.split())} words, {len(tokenizer(ROUTER_SYSTEM).input_ids) if tokenizer else '?'} tokens.\n")
print("""  This prompt states the rules in full, which makes the baseline FAIR: the
  base model is not being asked to guess a policy it has never seen.

  The same prompt is used in PART 4's training examples and in PART 5's
  measurement. One prompt everywhere, so the only thing that changes between
  the before and after numbers is the weights.

  It is also why A2 matters here. This prompt is identical in all 240 training
  examples, so unmasked training would spend most of its gradient learning to
  recite it. PART 4 masks it.
""")


print("=" * 72)
print("PART 3.2  THE GRADER")
print("=" * 72)

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.S)


def extract_json(text):
    """Return (obj_or_None, needed_repair). Repair means we had to strip or
    hunt for the object -- a model that needs it did not follow the prompt."""
    try:
        return json.loads(text), False
    except Exception:
        pass
    stripped = FENCE.sub("", text)
    try:
        return json.loads(stripped), True
    except Exception:
        pass
    m = re.search(r"\{.*\}", stripped, re.S)          # greedy: outermost braces
    if m:
        try:
            return json.loads(m.group(0)), True
        except Exception:
            pass
    return None, True


def norm(v):
    """Make comparison fair without making it loose."""
    if isinstance(v, str):
        v = v.strip()
        if v.lower() in ("", "null", "none"):
            return None
        if v.lstrip("-").isdigit():                  # "5" and 5 are the same answer
            return int(v)
    return v


def grade(replies, rows):
    """Four metrics. Every denominator is len(rows), including the unparseable."""
    n = len(rows)
    score = {"parse_rate": 0, "no_repair": 0, "schema_valid": 0, "exact_match": 0}
    per_field = {f: 0 for f in FIELDS}
    failures = []

    for raw, row in zip(replies, rows):
        obj, repaired = extract_json(raw)
        if obj is None or not isinstance(obj, dict):
            failures.append((row["message"], raw, "no JSON"))
            continue
        score["parse_rate"] += 1
        score["no_repair"] += (not repaired)
        if not sorted(VALIDATOR.iter_errors(obj), key=str):
            score["schema_valid"] += 1
        hits = 0
        for f in FIELDS:
            if norm(obj.get(f)) == norm(row["gold"][f]):
                per_field[f] += 1
                hits += 1
        if hits == len(FIELDS) and set(obj) == set(FIELDS):
            score["exact_match"] += 1
        elif len(failures) < 3:
            wrong = [f for f in FIELDS if norm(obj.get(f)) != norm(row["gold"][f])]
            failures.append((row["message"], json.dumps(obj), "wrong: " + ", ".join(wrong)))

    out = {k: v / n * 100 for k, v in score.items()}
    out.update({f: per_field[f] / n * 100 for f in FIELDS})
    out["_n"] = n
    out["_failures"] = failures
    return out


def report(res, label):
    n = res["_n"]
    print(f"  {label}   (n = {n})\n")
    for k, note in [("parse_rate", "replied with parseable JSON"),
                    ("no_repair", "...without needing a repair"),
                    ("schema_valid", "every field present and legal"),
                    ("exact_match", "all 7 fields correct")]:
        print(f"    {k:<14} {res[k]:>6.1f}%    {note}")
    print(f"\n    {'per field':<22} {'correct':>8}   type")
    for f in FIELDS:
        kind = "RULE" if f in RULE_FIELDS else ("class" if f == "intent" else "extract")
        print(f"    {f:<22} {res[f]:>7.1f}%   {kind}")
    if res["_failures"]:
        print("\n    examples that went wrong:")
        for msg, got, why in res["_failures"][:3]:
            print(f"      {msg[:62]}")
            print(f"        {why}")
            print(f"        got: {got[:90]}")
    print()


print("""  Four metrics, in order of how little they demand:

    parse_rate     did a JSON object come back at all
    no_repair      came back clean, with no markdown fence to strip
    schema_valid   all seven fields, each a legal value
    exact_match    all seven fields correct

  Plus one accuracy per field, because the fields fail for different reasons
  and the average hides that. No LLM judge is involved anywhere: every
  comparison is ==, which is only possible because PART 2's labels came from
  a rule.
""")


print("=" * 72)
print("PART 3.3  THE BASE MODEL'S SCORE")
print("=" * 72)


def generate_batch(rows, max_new_tokens=160, bs=8, tag=""):
    """Batched greedy generation. Left padding, because generation continues
    from the RIGHT edge and right padding would put pad tokens there."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    outs = []
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        texts = [tokenizer.apply_chat_template(
            [{"role": "system", "content": ROUTER_SYSTEM},
             {"role": "user", "content": r["message"]}],
            tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        outs += tokenizer.batch_decode(new, skip_special_tokens=True)
        print(f"  {tag}generating {len(outs)}/{len(rows)}", end="\r")
    print(f"  {tag}generated {len(outs)}/{len(rows)}      ")
    return outs


base_replies = generate_batch(test, tag="base: ")
before = grade(base_replies, test)
report(before, "Qwen2.5-1.5B-Instruct, no fine-tuning")

print("""  Compare the rows, not the headline.

  parse_rate and schema_valid are usually high, because Qwen's post-training
  already taught it to follow a format instruction. The rule fields are where
  it falls down, and priority is the worst of the three, because it is the
  only field that needs three facts combined in a fixed order.

  That gap is the thing a fine-tune can close, and it is why this project
  picked a rule-based schema instead of a one-field label.
""")

print("=" * 72)
print("END OF PART 3.  Next: PART 4 trains the adapter.")
print("=" * 72)


# ===========================================================================
# PART 4 OF 5 -- THE FINE-TUNE
# ===========================================================================
# What this part does:
#   1. attaches LoRA with peft, and reports what fraction is trainable   (A4)
#   2. encodes every example and masks the prompt by length              (A2, A6)
#   3. checks the prefix assumption on all 240 examples, not just one
#   4. trains with A3's four lines, plus the loss scaling fp16 needs
#   5. saves the adapter, which is the only artefact worth keeping
# ---------------------------------------------------------------------------

from peft import LoraConfig, get_peft_model

RANK, ALPHA, DROPOUT = 16, 32, 0.05
EPOCHS, LR, BS = 3, 2e-4, 4
MAX_LEN = 640
ADAPTER_DIR = "qwen-router-lora"

print("=" * 72)
print("PART 4.1  ATTACHING LoRA")
print("=" * 72)

before_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

lora = LoraConfig(
    r=RANK,
    lora_alpha=ALPHA,
    lora_dropout=DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    # Qwen2 names its attention projections separately, unlike GPT-2's fused
    # c_attn. So four target modules here, where A4 wrapped one.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(model, lora)
model.config.use_cache = False               # incompatible with training; saves memory

# fp16 is too coarse for the optimizer's own arithmetic. Keep the trainable
# tensors in fp32; peft casts activations into their dtype on the way through.
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()

total = sum(p.numel() for p in model.parameters())
train_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)

print(f"  trainable before      : {before_trainable:,}")
print(f"  trainable now         : {train_n:,}   ({train_n / total * 100:.2f}% of {total:,})")
print(f"  new tensors added     : {n_lora}   (one A and one B per target module per layer)")
print(f"  r = {RANK}, alpha = {ALPHA}, scale = alpha/r = {ALPHA / RANK}")
print(f"  targets               : q_proj, k_proj, v_proj, o_proj\n")

print(f"""  A4 wrapped GPT-2's c_attn, one module per block, because GPT-2 fuses Q, K
  and V into a single (768, 2304) matrix. Qwen2 keeps them separate, so the
  same idea needs four target modules instead of one.

  Memory, measured, not estimated:

    weights (frozen, 16-bit)   : {torch.cuda.memory_allocated() / 1e9:.2f} GB
    gradients, 1 per trainable : {train_n * 4 / 1e9:.4f} GB
    AdamW state, 2 per         : {train_n * 8 / 1e9:.4f} GB

  A full fine-tune would need {total * 4 * 3 / 1e9:.1f} GB for those last two rows alone.
  That is the whole reason LoRA exists, and it is A3's arithmetic unchanged.
""")


print("=" * 72)
print("PART 4.2  ENCODING, AND THE MASK")
print("=" * 72)


def encode(row):
    """Return (input_ids, labels). labels is -100 everywhere the prompt is."""
    msgs = [{"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": row["message"]}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        msgs + [{"role": "assistant", "content": json.dumps(row["gold"])}],
        tokenize=False)
    p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    f_ids = tokenizer(full, add_special_tokens=False).input_ids[:MAX_LEN]
    labels = list(f_ids)
    labels[:len(p_ids)] = [-100] * min(len(p_ids), len(labels))
    return f_ids, labels, len(p_ids)


encoded, bad_prefix = [], 0
for row in train:
    msgs = [{"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": row["message"]}]
    p_ids = tokenizer(tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True),
        add_special_tokens=False).input_ids
    f_ids, labels, plen = encode(row)
    if f_ids[:len(p_ids)] != p_ids:          # A6's check, on every example
        bad_prefix += 1
    encoded.append((f_ids, labels))

graded = sum(sum(1 for x in lab if x != -100) for _, lab in encoded)
tokens = sum(len(ids) for ids, _ in encoded)

print(f"  examples                      : {len(encoded)}")
print(f"  prefix check failures         : {bad_prefix}   (must be 0)")
print(f"  total tokens                  : {tokens:,}")
print(f"  graded tokens                 : {graded:,}")
print(f"  share of tokens graded        : {graded / tokens * 100:.1f}%\n")

ids0, lab0 = encoded[0]
print(f"  example 0: {len(ids0)} tokens, {sum(1 for x in lab0 if x != -100)} graded")
print(f"  the graded part decodes to:")
print(f"    {tokenizer.decode([x for x in lab0 if x != -100])!r}\n")

print(f"""  That {graded / tokens * 100:.1f}% is the number A2 was about.

  The system prompt is {len(tokenizer(ROUTER_SYSTEM).input_ids)} tokens and it is IDENTICAL in all {len(encoded)} examples.
  Without the mask, {100 - graded / tokens * 100:.1f}% of every gradient would go into learning to
  recite a string we hand the model at inference time anyway. The loss would
  still fall. The model would still train. It would just be worse.

  Notice what IS graded: the JSON, and the <|im_end|> that ends the turn. That
  last token is how the model learns to stop.
""")


print("=" * 72)
print("PART 4.3  TRAINING")
print("=" * 72)


def collate(batch):
    """Right padding. Pad tokens get attention 0 and label -100, so they are
    neither read nor graded."""
    width = max(len(ids) for ids, _ in batch)
    pad = tokenizer.pad_token_id
    input_ids, labels, mask = [], [], []
    for ids, lab in batch:
        gap = width - len(ids)
        input_ids.append(ids + [pad] * gap)
        labels.append(lab + [-100] * gap)
        mask.append([1] * len(ids) + [0] * gap)
    return (torch.tensor(input_ids).to(model.device),
            torch.tensor(labels).to(model.device),
            torch.tensor(mask).to(model.device))


optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=LR)

# fp16 has a narrow window: it underflows below 6.1e-05. A gradient smaller
# than that becomes zero and the weight never moves. The scaler multiplies the
# loss up before backward and divides the gradients back down afterwards.
# bf16 has fp32's range, so on an Ampere GPU this is a no-op.
scaler = torch.amp.GradScaler("cuda", enabled=not BF16)
print(f"  loss scaling          : {'ON (fp16)' if not BF16 else 'off (bf16 does not need it)'}")
print(f"  epochs {EPOCHS}, batch size {BS}, lr {LR}, "
      f"{EPOCHS * ((len(encoded) + BS - 1) // BS)} steps\n")

order = list(range(len(encoded)))
rng_shuffle = random.Random(7)
step, t_start = 0, time.time()
print(f"  {'step':>5} {'loss':>9}   bar")
for epoch in range(EPOCHS):
    rng_shuffle.shuffle(order)
    model.train()
    for i in range(0, len(order), BS):
        batch = [encoded[j] for j in order[i:i + BS]]
        input_ids, labels, mask = collate(batch)

        loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
        scaler.scale(loss).backward()       # A3's backward, with scaling
        scaler.step(optimizer)              # A3's step
        scaler.update()
        optimizer.zero_grad()               # A3's zero_grad

        step += 1
        if step % 15 == 0 or step == 1:
            v = loss.item()
            print(f"  {step:>5} {v:>9.4f}   {'#' * int(min(v, 6) * 8)}")

print(f"\n  trained {step} steps in {time.time() - t_start:.0f} s")
print(f"  peak VRAM             : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

print("""
  Four lines did the work:

      loss = model(input_ids, attention_mask, labels).loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

  Identical to A3, on a model 12x larger, with 0.5% of the weights moving. The
  scaler wrapped around them is fp16 bookkeeping, not a fifth idea.
""")


print("=" * 72)
print("PART 4.4  SAVING THE ADAPTER")
print("=" * 72)

model.save_pretrained(ADAPTER_DIR)
files = sorted(os.listdir(ADAPTER_DIR))
size = sum(os.path.getsize(os.path.join(ADAPTER_DIR, f)) for f in files)

print(f"  {ADAPTER_DIR}/")
for f in files:
    print(f"    {f:<34} {os.path.getsize(os.path.join(ADAPTER_DIR, f)):>12,} bytes")
print(f"\n  total                              {size:>12,} bytes"
      f"   ({size / 1e6:.1f} MB)")
print(f"  the base model, for comparison     {total * 2:>12,} bytes"
      f"   ({total * 2 / 1e9:.2f} GB)")
print(f"  ratio                              {total * 2 / size:>12.0f}x smaller")

print(f"""
  Download that folder, not the model. The base weights are a public 3.1 GB
  file anyone can refetch from Hugging Face; your {size / 1e6:.0f} MB adapter is the only
  thing this run produced that did not exist before.

  Colab disconnects without warning. If this run mattered, copy the folder to
  Drive now:

      from google.colab import drive; drive.mount('/content/drive')
      !cp -r {ADAPTER_DIR} /content/drive/MyDrive/
""")

print("=" * 72)
print("END OF PART 4.  Next: PART 5 scores the fine-tune against PART 3.")
print("=" * 72)


# ===========================================================================
# PART 5 OF 5 -- THE VERDICT
# ===========================================================================
# What this part does:
#   1. scores the fine-tune on the SAME 60 examples, with the SAME grader
#   2. prints before and after side by side, per field
#   3. asks whether it learned the RULE or just the easy half of it
#   4. states the limits, and shows how to load the adapter again later
#
# No training here. Nothing is saved. This part only measures.
# ---------------------------------------------------------------------------

print("=" * 72)
print("PART 5.1  THE FINE-TUNE'S SCORE")
print("=" * 72)

model.config.use_cache = True                 # back on: generation wants the KV cache
tuned_replies = generate_batch(test, tag="tuned: ")
after = grade(tuned_replies, test)
report(after, "Qwen2.5-1.5B-Instruct + LoRA")


print("=" * 72)
print("PART 5.2  BEFORE AND AFTER")
print("=" * 72)

print(f"  {'metric':<22} {'base':>8} {'tuned':>8} {'change':>10}   type")
print(f"  {'-' * 22} {'-' * 8} {'-' * 8} {'-' * 10}   {'-' * 7}")
for k in ["parse_rate", "no_repair", "schema_valid", "exact_match"]:
    print(f"  {k:<22} {before[k]:>7.1f}% {after[k]:>7.1f}% "
          f"{after[k] - before[k]:>+9.1f}pp")
print(f"  {'-' * 22} {'-' * 8} {'-' * 8} {'-' * 10}   {'-' * 7}")
for f in FIELDS:
    kind = "RULE" if f in RULE_FIELDS else ("class" if f == "intent" else "extract")
    print(f"  {f:<22} {before[f]:>7.1f}% {after[f]:>7.1f}% "
          f"{after[f] - before[f]:>+9.1f}pp   {kind}")

print(f"""
  n = {len(test)}, so one example is {100 / len(test):.1f} percentage points. Anything smaller than
  about {300 / len(test):.0f}pp is noise, not a result.

  Read the three RULE rows first. Those are the fields a prompt alone was
  asking the model to compute, and they are what a fine-tune is actually for.
  The extract rows measure copying, which Qwen could already do.
""")

print("  the same message, before and after:\n")
for i in (0, 1):
    print(f"    {test[i]['message'][:68]}")
    print(f"      base : {base_replies[i].strip()[:86]!r}")
    print(f"      tuned: {tuned_replies[i].strip()[:86]!r}")
    print(f"      want : {json.dumps(test[i]['gold'])[:86]!r}\n")


print("=" * 72)
print("PART 5.3  DID IT LEARN THE RULE, OR HALF OF IT?")
print("=" * 72)


def subset(rows, replies, field, pick):
    """Accuracy on `field`, over the rows where pick(row) is true."""
    hit = n = 0
    for raw, row in zip(replies, rows):
        if not pick(row):
            continue
        n += 1
        obj, _ = extract_json(raw)
        if isinstance(obj, dict) and norm(obj.get(field)) == norm(row["gold"][field]):
            hit += 1
    return (hit / n * 100 if n else float("nan")), n


HYG_LOOKS = ("tooth", "wash", "shav", "cotton bud", "towel", "soap")


def moved(row):
    """True if RULE 3 moved the priority away from the intent's base level."""
    g = row["gold"]
    return g["priority"] != priority(g["intent"], None, False)


print("  Two questions the headline number cannot answer.\n")
print(f"  {'subset':<40} {'base':>7} {'tuned':>7} {'n':>4}")
print(f"  {'-' * 40} {'-' * 7} {'-' * 7} {'-' * 4}")

for label, field, pick in [
    ("is_hygiene_item, word IS on the list",
     "is_hygiene_item", lambda r: r["gold"]["is_hygiene_item"] is True),
    ("is_hygiene_item, word NOT on the list",
     "is_hygiene_item", lambda r: (not r["gold"]["is_hygiene_item"]) and any(
         w in r["gold"]["item_name"].lower() for w in HYG_LOOKS)),
    ("priority, no adjustment applied",
     "priority", lambda r: not moved(r)),
    ("priority, an adjustment moved it",
     "priority", moved),
]:
    b, n = subset(test, base_replies, field, pick)
    t, _ = subset(test, tuned_replies, field, pick)
    print(f"  {label:<40} {b:>6.1f}% {t:>6.1f}% {n:>4}")

print("""
  Row 1 against row 2 is the hygiene trap. Row 1 asks the model to apply the
  list. Row 2 asks it to NOT apply the idea: toothpaste is obviously a hygiene
  product, and the correct answer is false, because the word is not on the
  list. A model that scores well on row 1 and badly on row 2 learned the
  concept instead of the policy.

  Row 3 against row 4 splits priority the same way. Row 3 needs only the
  intent lookup. Row 4 needs the staleness and hygiene adjustments applied in
  the right order. If row 3 is high and row 4 is low, the model learned the
  base table and ignored the adjustments -- which would score about 60% on the
  field overall and look like progress.
""")


print("=" * 72)
print("PART 5.4  WHAT THIS DOES AND DOES NOT SHOW")
print("=" * 72)

print(f"""  What it shows.

  The test set was held out along the two dimensions that vary in production:
  the ITEMS and the PHRASING register. Neither appears in training. That is
  the correction A7 earned -- there, the test set shared its phrasings with
  training, scored 100%, and collapsed to 33% on new wording.

  What it does not show.

  {len(test)} examples. One example is {100 / len(test):.1f}pp. A 5pp difference is three examples.

  The messages are machine-written from a spec. Real customers misspell,
  mention two problems at once, and leave out the order id entirely. None of
  that is in this dataset, and the score here will be optimistic because of
  it.

  Only {len(train)} training examples. If the rule fields are still weak, more varied
  data is the first thing to try -- before a bigger rank, more epochs, or a
  bigger model.

  Using the adapter later.

      from peft import PeftModel
      base = AutoModelForCausalLM.from_pretrained(
          "{MODEL}", dtype=torch.float16, device_map="cuda")
      model = PeftModel.from_pretrained(base, "{ADAPTER_DIR}")

  That loads 3.1 GB of public weights plus your {sum(os.path.getsize(os.path.join(ADAPTER_DIR, f)) for f in os.listdir(ADAPTER_DIR)) / 1e6:.0f} MB. To collapse the two
  into one set of weights -- W + (alpha/r) * B @ A, done by hand -- that is
  the serving track's S2.
""")

print("=" * 72)
print("DONE.  Five parts: model, data, grader, fine-tune, verdict.")
print("=" * 72)
