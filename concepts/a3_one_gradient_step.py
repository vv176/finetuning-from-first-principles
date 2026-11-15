"""
A3: Training is a loop around A1's loss. That is the whole algorithm.

A1 computed a loss. A2 controlled which positions it covers. Neither changed
the model. This file changes it.

The loop is four lines:

    loss = model(input_ids, labels).loss     # A1: how wrong are we
    loss.backward()                          # for every weight: which way is downhill
    optimizer.step()                         # take one small step that way
    optimizer.zero_grad()                    # clear the gradients, or they accumulate

Everything else in a real training script -- batching, schedulers, checkpoints,
LoRA, 4-bit -- is support around those four lines. None of it changes what they
do.

Two warnings this file pays off.

A1 and A2 called model.eval() with a note saying A3 explains it. Section 1 does.

Section 5 shows what 20 steps on ONE example actually buys. The answer is: a
model that memorised one answer and now gives it to every question. Seeing that
early is the point. It is why the project has a held-out test set instead of
trusting a falling loss.

Run it:
    python3 a3_one_gradient_step.py

Measured on an M1 with 8 GB: 30 to 48 seconds depending on whether the weights
are already in the OS disk cache. Peak resident 2.4 GB, peak memory footprint
3.6 GB -- higher than A1 and A2 because this file keeps a spare copy of the
weights so section 4 can reset. Close other apps before running it.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(0)                # so the dropout numbers below reproduce

MODEL = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)

PROMPT = 'Customer message: my bottle arrived cracked\nJSON:'
ANSWER = ' {"intent": "damaged_item"}'
OTHER = 'Customer message: where is my order\nJSON:'

prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
input_ids = tokenizer(PROMPT + ANSWER, return_tensors="pt").input_ids
labels = input_ids.clone()
labels[:, :prompt_ids.shape[1]] = -100          # A2: grade the answer only

# Keep a pristine copy so section 4 can start from the untouched model.
original = {k: v.clone() for k, v in model.state_dict().items()}


def loss_now():
    """A1's forward pass. eval() mode, no gradients, so it is a pure measurement."""
    model.eval()
    with torch.no_grad():
        return model(input_ids=input_ids, labels=labels).loss.item()


def generate(text, n=14):
    model.eval()
    ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=n, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][ids.shape[1]:])


# ---------------------------------------------------------------------------
# 1. train() and eval(). The warning from A1 and A2, paid off.
# ---------------------------------------------------------------------------
print("=" * 72)
print("1. WHY A1 AND A2 CALLED model.eval()")
print("=" * 72)
print("""  A PyTorch model has two modes, and they compute different things.

  DROPOUT is the reason. During training, dropout randomly sets a fraction of
  the internal values to zero on every forward pass. It stops the network
  leaning too hard on any single path through itself, which reduces
  overfitting. GPT-2 drops 10%.

  That randomness happens in train() mode only. eval() turns it off.

  Same input, same weights, two forward passes each way:
""")

model.train()
with torch.no_grad():
    for k in range(2):
        l = model(input_ids=input_ids, labels=labels).loss.item()
        print(f"    train() run {k + 1} : loss {l:.4f}")

model.eval()
with torch.no_grad():
    for k in range(2):
        l = model(input_ids=input_ids, labels=labels).loss.item()
        print(f"    eval()  run {k + 1} : loss {l:.4f}")

print("""
  train() gives two different numbers. eval() gives the same number twice.
  Nothing about the model changed between any of those four calls.

  The rule: TRAIN in train(), MEASURE in eval(). If a loss moves when nothing
  changed, someone forgot to switch modes.
""")


# ---------------------------------------------------------------------------
# 2. One step, by hand. No optimizer.
# ---------------------------------------------------------------------------
print("=" * 72)
print("2. ONE GRADIENT STEP, BY HAND")
print("=" * 72)

LR = 1e-4

# Pick one weight to watch. There are 124 million of them; this is one.
watched = model.transformer.h[0].attn.c_attn.weight
print(f"  watching one weight inside {'transformer.h[0].attn.c_attn.weight'}")
print(f"  that tensor alone holds {watched.numel():,} numbers\n")

print(f"  loss before            : {loss_now():.6f}")
print(f"  weight before          : {watched[0, 0].item():+.8f}")
print(f"  gradient before        : {watched.grad}")

# backward() fills .grad for every parameter. It computes nothing else.
model.train()
loss = model(input_ids=input_ids, labels=labels).loss
loss.backward()

g = watched.grad[0, 0].item()
print(f"\n  after loss.backward():")
print(f"  gradient               : {g:+.8f}")
print(f"  what it means          : raising this weight by 1.0 would change the")
print(f"                           loss by about {g:+.6f}")

# The update. This single line is what an optimizer does, in its simplest form.
with torch.no_grad():
    for p in model.parameters():
        if p.grad is not None:
            p -= LR * p.grad                    # plain SGD: step downhill

print(f"\n  applied  w = w - {LR} * grad   (plain SGD, every parameter)")
print(f"  weight after           : {watched[0, 0].item():+.8f}")
print(f"  change                 : {-LR * g:+.10f}")
print(f"  loss after             : {loss_now():.6f}")

print("""
  The loss went down. That is the entire mechanism: ask which way is downhill,
  move a small distance that way, repeat.

  Now put the two changes side by side, because this is the part that surprises
  people. The watched weight moved by 0.0000002. The loss moved by 0.27.

  One weight barely twitched. The loss fell because ALL 124 million twitched at
  once, each in its own downhill direction, and the effects added up. Gradient
  descent is not a few weights doing something big; it is every weight doing
  something almost too small to see.
""")

model.zero_grad()


# ---------------------------------------------------------------------------
# 3. The same step, with the API.
# ---------------------------------------------------------------------------
print("=" * 72)
print("3. THE SAME STEP, WITH torch.optim")
print("=" * 72)
print("""  Section 2 hand-wrote the update rule. In practice you hand the parameters
  to an optimizer and call step(). AdamW is what fine-tuning scripts use: it
  keeps a running average of each weight's gradient and of its square, so
  weights with consistently small gradients still move.

  The loop is the same four lines either way:

      loss = model(input_ids=ids, labels=labels).loss
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

  Those running averages are not free. AdamW stores TWO extra numbers per
  weight, for the whole model, for the whole run:
""")

n_params = sum(p.numel() for p in model.parameters())
bytes_per = 4                                   # fp32
gb = n_params * bytes_per / 1e9
print(f"    {'weights':<28}: {gb:.2f} GB   ({n_params:,} x 4 bytes)")
print(f"    {'gradients, 1 per weight':<28}: {gb:.2f} GB")
print(f"    {'AdamW state, 2 per weight':<28}: {2 * gb:.2f} GB")
print(f"    {'-' * 28}  {'-' * 7}")
print(f"    {'total, just to train':<28}: {4 * gb:.2f} GB")

print(f"""
  So training needs roughly 4x the memory of the weights alone. On GPT-2 that
  is fine. Scale it to the 1.5B model the project uses and the same arithmetic
  gives about {4 * 1.5e9 * 4 / 1e9:.0f} GB, which does not fit a free Colab GPU.

  That single number is why LoRA (A4) and 4-bit weights (A5) exist.
""")


# ---------------------------------------------------------------------------
# 4. Twenty steps.
# ---------------------------------------------------------------------------
print("=" * 72)
print("4. TWENTY STEPS ON ONE EXAMPLE")
print("=" * 72)

model.load_state_dict(original)         # back to untouched GPT-2
print("  reset to the original weights, so the curve starts clean\n")

print(f"  before training, the model answers:")
print(f"    {PROMPT!r}\n      -> {generate(PROMPT)!r}")
print(f"    {OTHER!r}\n      -> {generate(OTHER)!r}")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

print(f"\n  {'step':>5} {'loss':>10}   bar")
for step in range(1, 21):
    model.train()
    loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    v = loss.item()
    print(f"  {step:>5} {v:>10.4f}   {'#' * int(v * 6)}")


# ---------------------------------------------------------------------------
# 5. What it learned. Not what you want.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("5. WHAT 20 STEPS ON ONE EXAMPLE ACTUALLY BOUGHT")
print("=" * 72)

print(f"  final loss on the training example : {loss_now():.6f}\n")
print(f"  the message it trained on:")
print(f"    {PROMPT!r}\n      -> {generate(PROMPT)!r}")
print(f"\n  a DIFFERENT message:")
print(f"    {OTHER!r}\n      -> {generate(OTHER)!r}")

print("""
  Read those two answers together.

  On the trained message it is perfect. On a message it has never seen it
  answers 'damaged_item' as well, which is wrong -- that one is a missing
  delivery, not a damaged item.

  It did not learn the task. It learned one answer, and now it gives that
  answer to everything. The name for this is OVERFITTING: loss on the
  training data goes to zero while the model gets worse at everything else.

  This is why a falling loss curve proves nothing on its own, and why the
  project measures on a held-out test set the model never trained on.
""")


# ---------------------------------------------------------------------------
# 6. What separates this from a real fine-tune.
# ---------------------------------------------------------------------------
print("=" * 72)
print("6. WHAT A REAL FINE-TUNE ADDS")
print("=" * 72)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
rows = [
    ("1 example",                "~1000 examples, shuffled"),
    ("20 steps",                 "several hundred steps"),
    ("no batching",              "batches, padded, with an attention mask"),
    ("no validation",            "held-out loss checked as it trains"),
    (f"all {trainable / 1e6:.0f}M weights trainable", "~1% trainable (LoRA -- A4)"),
    ("fp32 weights",             "4-bit base weights (QLoRA -- A5)"),
]

print("  Mechanically, nothing. Those same four lines run in the Colab notebook.")
print("  What changes is scale and bookkeeping:\n")
print(f"    {'this file':<30}  {'a real run'}")
print(f"    {'-' * 30}  {'-' * 38}")
for a, b in rows:
    print(f"    {a:<30}  {b}")

print(f"""
  The last two rows exist for one reason. {trainable / 1e6:.0f}M weights was fine on a
  laptop. A 1.5B model needs the {4 * 1.5e9 * 4 / 1e9:.0f} GB from section 3, and a free Colab
  T4 has 16 GB. A4 and A5 are how you close that gap without changing the
  four lines.
""")
