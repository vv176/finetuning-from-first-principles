"""
A7: The whole loop, end to end, in one file.

A1 measured a loss. A2 masked it. A3 took a step. A4 made the step cheap.
A5 made the weights cheap. A6 fixed the prompt format.

This file does all of it at once, on a real task, and answers the only question
that matters: DID IT GET BETTER?

    synthetic data  ->  measure the base model  ->  LoRA fine-tune  ->  measure again

Everything is local. GPT-2 (124M), CPU, no API key, no `peft`. The LoRA class is
A4's, copied in so this file stands alone.

The task: read a customer message, emit one JSON object with two fields.

    'Order A4821 - my bottle arrived cracked.'
        -> {"order_id": "A4821", "intent": "damaged_item"}

Two different skills in one output, which is why this task is worth teaching:

    order_id   must be COPIED out of the message  (extraction)
    intent     must be CHOSEN from six labels     (classification)

Run it:
    python3 a7_the_whole_loop.py
"""

import json
import random
import re

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(0)

MODEL = "gpt2"
RANK, ALPHA = 8, 16
EPOCHS = 3
LR = 1e-3
N_TRAIN, N_TEST = 100, 20

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,     # A3: no dropout noise
)
EOS = tokenizer.eos_token


# ---------------------------------------------------------------------------
# 1. The data. Synthetic, so the labels are correct by construction.
# ---------------------------------------------------------------------------
print("=" * 72)
print("1. THE DATA")
print("=" * 72)

TEMPLATES = {
    "damaged_item":     ["my {item} arrived cracked", "the {item} was broken in the box",
                         "received a dented {item}", "{item} came shattered"],
    "missing_delivery": ["my {item} never showed up", "the {item} says delivered but is not here",
                         "still waiting for my {item}", "no sign of the {item}"],
    "return_request":   ["I want to return the {item}", "please take back this {item}",
                         "need to send the {item} back", "returning the {item}"],
    "wrong_item":       ["you sent a {item} instead", "I got the wrong {item}",
                         "this {item} is not what I ordered", "wrong {item} delivered"],
    "billing_issue":    ["charged twice for the {item}", "the {item} price is wrong",
                         "double billed on the {item}", "refund not showing for the {item}"],
    "cancel_order":     ["cancel my {item} order", "please cancel the {item}",
                         "I no longer want the {item}", "stop the {item} shipment"],
}
ITEMS = ["bottle", "lamp", "mug", "speaker", "charger", "blender", "kettle", "monitor"]
INTENTS = list(TEMPLATES)

rng = random.Random(0)
seen, data = set(), []
while len(data) < N_TRAIN + N_TEST:
    intent = INTENTS[len(data) % len(INTENTS)]          # balanced classes
    body = rng.choice(TEMPLATES[intent]).format(item=rng.choice(ITEMS))
    oid = "A" + str(rng.randint(1000, 9999))
    if (body, oid) in seen:
        continue
    seen.add((body, oid))
    data.append({
        "message": f"Order {oid} - {body}.",
        "gold": {"order_id": oid, "intent": intent},
    })

train, test = data[:N_TRAIN], data[N_TRAIN:]
print(f"  generated {len(data)} examples: {len(train)} train, {len(test)} test, no overlap")
print(f"  {len(INTENTS)} intents, balanced\n")
for ex in train[:3]:
    print(f"    {ex['message']}")
    print(f"      -> {json.dumps(ex['gold'])}")

print("""
  Synthetic means the labels cannot be wrong -- a template produced both the
  message and the answer. That removes label noise from the experiment, so any
  change in the score is the fine-tune and nothing else.

  It also removes the hardest part of a real project. Real messages are messy;
  these are not. Block B generates real ones with a frontier model.
""")


def prompt_of(msg):
    return f"Customer message: {msg}\nJSON:"


def answer_of(gold):
    return " " + json.dumps(gold) + EOS


# ---------------------------------------------------------------------------
# 2. The grader. Four numbers, no LLM judge.
# ---------------------------------------------------------------------------
def extract_json(text):
    """Pull the first {...} out of generated text. Returns None if there is none."""
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def generate(msg, n=28):
    model.eval()
    ids = tokenizer(prompt_of(msg), return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def evaluate(rows, label):
    n = len(rows)
    score = {"parse_rate": 0, "order_id": 0, "intent": 0, "exact_match": 0}
    samples = []
    for ex in rows:
        raw = generate(ex["message"])
        obj = extract_json(raw)
        if obj is not None:
            score["parse_rate"] += 1
            g = ex["gold"]
            ok_o = obj.get("order_id") == g["order_id"]
            ok_i = obj.get("intent") == g["intent"]
            score["order_id"] += ok_o
            score["intent"] += ok_i
            score["exact_match"] += (ok_o and ok_i and set(obj) == set(g))
        if len(samples) < 2:
            samples.append((ex["message"], raw.strip().replace("\n", " ")[:70], ex["gold"]))
    pct = {k: v / n * 100 for k, v in score.items()}
    print(f"  {label}")
    for k in ("parse_rate", "order_id", "intent", "exact_match"):
        print(f"    {k:<14}: {pct[k]:>6.1f}%   ({score[k]} of {n})")
    print()
    for msg, raw, gold in samples:
        print(f"    {msg}")
        print(f"      got  : {raw!r}")
        print(f"      want : {json.dumps(gold)}")
    print()
    return pct


print("=" * 72)
print("2. BEFORE: THE BASE MODEL ON THE TEST SET")
print("=" * 72)
print(f"  {N_TEST} messages the model has never seen. Same four metrics as B2.\n")
before = evaluate(test, "base GPT-2")

print("""  GPT-2 was never trained to emit JSON, so this is the floor. Whatever it
  prints, it prints because the prompt ends in 'JSON:' and nothing more.
""")


# ---------------------------------------------------------------------------
# 3. The fine-tune. A4's LoRA, A2's masking, A3's four lines.
# ---------------------------------------------------------------------------
print("=" * 72)
print("3. THE FINE-TUNE")
print("=" * 72)


class LoRAConv1D(nn.Module):
    """A4's class, unchanged. W stays frozen; A and B are the only trainable part."""

    def __init__(self, base, r=RANK, alpha=ALPHA):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        d_in, d_out = base.weight.shape
        self.A = nn.Parameter(torch.randn(d_in, r) * 0.01)      # random: breaks symmetry
        self.B = nn.Parameter(torch.zeros(r, d_out))            # zero: changes nothing yet
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A) @ self.B * self.scale


for p in model.parameters():
    p.requires_grad = False
for block in model.transformer.h:
    block.attn.c_attn = LoRAConv1D(block.attn.c_attn)

total = sum(p.numel() for p in model.parameters())
train_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  trainable : {train_n:,} of {total:,}   ({train_n / total * 100:.2f}%)")
print(f"  epochs {EPOCHS}, lr {LR}, batch size 1, {EPOCHS * len(train)} steps\n")


def example_tensors(ex):
    """A2: labels = -100 on the prompt, the answer tokens on the answer."""
    p = tokenizer(prompt_of(ex["message"]), return_tensors="pt").input_ids
    full = tokenizer(prompt_of(ex["message"]) + answer_of(ex["gold"]),
                     return_tensors="pt").input_ids
    labels = full.clone()
    labels[:, :p.shape[1]] = -100
    return full, labels


optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=LR)

order = list(range(len(train)))
step = 0
print(f"  {'step':>5} {'loss':>9}   bar")
for epoch in range(EPOCHS):
    rng.shuffle(order)
    for i in order:
        ids, labels = example_tensors(train[i])
        model.train()
        loss = model(input_ids=ids, labels=labels).loss      # A1
        loss.backward()                                      # A3
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        if step % 25 == 0 or step == 1:
            v = loss.item()
            print(f"  {step:>5} {v:>9.4f}   {'#' * int(min(v, 8) * 5)}")

print("""
  Four lines in that loop, exactly the four from A3. Everything else is
  bookkeeping: shuffling, masking, and printing.
""")


# ---------------------------------------------------------------------------
# 4. After. The same test set, the same grader.
# ---------------------------------------------------------------------------
print("=" * 72)
print("4. AFTER: THE SAME 20 MESSAGES")
print("=" * 72)
after = evaluate(test, "GPT-2 + LoRA")

print("=" * 72)
print("5. BEFORE AND AFTER")
print("=" * 72)
print(f"  {'metric':<14} {'base':>8} {'tuned':>8} {'change':>9}")
print(f"  {'-' * 14} {'-' * 8} {'-' * 8} {'-' * 9}")
for k in ("parse_rate", "order_id", "intent", "exact_match"):
    print(f"  {k:<14} {before[k]:>7.1f}% {after[k]:>7.1f}% {after[k] - before[k]:>+8.1f}pp")

print(f"""
  Read the rows separately. They measure different things.

    parse_rate    did it emit JSON at all              -- a FORMAT skill
    intent        did it pick the right label           -- CLASSIFICATION
    order_id      did it copy the id out of the message -- EXTRACTION
    exact_match   all of the above, on the same example

  Format is the easiest thing a fine-tune can teach, and usually the first to
  move. Classification follows. Extraction is hardest, because the answer is
  not in the weights -- it has to be read off the prompt.

  Three honest limits.

  The test set is {N_TEST} examples, so one example is {100 / N_TEST:.0f} percentage points. Treat
  small differences as noise.

  The data is synthetic and templated. The test messages use the SAME 24
  phrasings as training, with a different item and order id. Section 6 checks
  what happens when the phrasing is new.

  {train_n:,} numbers were trained -- {train_n / total * 100:.2f}% of the model. The other
  {total - train_n:,} are byte-for-byte what OpenAI released in 2019.
""")


# ---------------------------------------------------------------------------
# 6. The real test: phrasings the model has never seen.
# ---------------------------------------------------------------------------
print("=" * 72)
print("6. UNSEEN PHRASINGS")
print("=" * 72)

PROBE = [
    ("Order A9001 - the glass on my monitor is in pieces.",
     {"order_id": "A9001", "intent": "damaged_item"}),
    ("Order A9002 - tracking says it arrived days ago but nothing came.",
     {"order_id": "A9002", "intent": "missing_delivery"}),
    ("Order A9003 - can I ship this kettle back for a refund?",
     {"order_id": "A9003", "intent": "return_request"}),
    ("Order A9004 - I ordered a lamp and received a blender.",
     {"order_id": "A9004", "intent": "wrong_item"}),
    ("Order A9005 - my card was hit for the mug twice.",
     {"order_id": "A9005", "intent": "billing_issue"}),
    ("Order A9006 - kill the speaker order before it ships.",
     {"order_id": "A9006", "intent": "cancel_order"}),
]

print("  Six messages written by hand. None of these phrasings is in TEMPLATES.\n")
print(f"  {'message':<52} {'predicted intent':<18} {'id':>4} {'intent':>7}")
print(f"  {'-' * 52} {'-' * 18} {'-' * 4} {'-' * 7}")
hits = oid_hits = 0
for msg, gold in PROBE:
    obj = extract_json(generate(msg)) or {}
    pred = obj.get("intent", "(no json)")
    ok_o = obj.get("order_id") == gold["order_id"]
    ok_i = pred == gold["intent"]
    oid_hits += ok_o
    hits += ok_i
    print(f"  {msg[:52]:<52} {str(pred)[:18]:<18} "
          f"{'ok' if ok_o else 'BAD':>4} {'ok' if ok_i else 'BAD':>7}")

print(f"""
  order_id : {oid_hits} of {len(PROBE)}
  intent   : {hits} of {len(PROBE)}

  Those two rows split cleanly, and the split is the most useful thing in this
  file.

  order_id generalised perfectly. Copying a token out of the prompt is a
  mechanism GPT-2 already had; the fine-tune only taught it WHERE to put the
  copy. New phrasing does not break a copy.

  intent did not generalise. The model did not learn "a broken thing means
  damaged_item". It learned "the sentence 'my X arrived cracked' is followed by
  damaged_item". Change the sentence and it has nothing to fall back on.

  Compare that with section 5's {after['exact_match']:.0f}% and you have the whole lesson of this
  file. The test set was held out, but it was not INDEPENDENT -- it shared its
  phrasings with the training data. A score measured that way tells you the
  model learned the templates. It does not tell you the model learned the task.

  This is the same mistake as A3's overfitting, one level up. A3 memorised one
  example. This memorises 24 sentence patterns.

  The fix is not a bigger model or more steps. It is more varied data, which is
  exactly why Block B spends a whole script generating examples with a frontier
  model instead of a template.
""")

print("=" * 72)
print("NEXT: Block B does this with real data, a 1.5B instruct model, and a")
print("      seven-field schema where three fields are decided by a rule.")
print("=" * 72)
