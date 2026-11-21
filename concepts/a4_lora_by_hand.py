"""
A4: LoRA. Train 0.24% of the weights and get the same job done.

A3 ended on a number. Training GPT-2 needs about 4x the memory of the weights
themselves, and the same arithmetic on a 1.5B model gives 24 GB. A free Colab
T4 has 16 GB. That gap is the only reason this file exists.

LoRA's claim: you do not need to update W at all.

Freeze W. Put two small matrices beside it, A and B, and learn those instead.
The layer computes

    out = x @ W  +  (x @ A) @ B * scale
          ^frozen    ^the only part that trains

W in GPT-2's attention is 768 x 2304. With rank r = 8, A is 768 x 8 and B is
8 x 2304. That is 24,576 numbers in place of 1,769,472 -- 72x fewer, same layer.

Two things have to be true before this is usable, and this file checks both.

  1. The base weights really never move. Section 3 compares them bit for bit
     after training.
  2. The model starts out identical to the base model. Section 2 shows why:
     B is initialised to ZERO, so (x @ A) @ B is zero on the first forward
     pass and the added term contributes nothing. Loss at step 0 is A1's
     3.609078, unchanged.

Then section 5 trains it and the loss falls the way A3's did.

Run it:
    python3 a4_lora_by_hand.py
"""

import copy

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(0)

MODEL = "gpt2"
RANK = 8
ALPHA = 16

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
)
# Dropout off, deliberately. A3 section 1 showed what it does and why
# train() and eval() disagree. Here it would only add noise to section 4's
# loss curve, and this file is about the optimizer, not about dropout.

PROMPT = 'Customer message: my bottle arrived cracked\nJSON:'
ANSWER = ' {"intent": "damaged_item"}'

prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
input_ids = tokenizer(PROMPT + ANSWER, return_tensors="pt").input_ids
labels = input_ids.clone()
labels[:, :prompt_ids.shape[1]] = -100          # A2: grade the answer only

TOTAL_PARAMS = sum(p.numel() for p in model.parameters())


def loss_now():
    """A1's forward pass. eval() mode, no gradients. A pure measurement."""
    model.eval()
    with torch.no_grad():
        return model(input_ids=input_ids, labels=labels).loss.item()


def trainable():
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
# 1. The problem, in the numbers from A3.
# ---------------------------------------------------------------------------
print("=" * 72)
print("1. WHY FULL FINE-TUNING DOES NOT FIT")
print("=" * 72)

gb = TOTAL_PARAMS * 4 / 1e9
print(f"  GPT-2 has {TOTAL_PARAMS:,} parameters, and A3 trained every one of them.\n")
print(f"    {'weights':<28}: {gb:.2f} GB")
print(f"    {'gradients':<28}: {gb:.2f} GB")
print(f"    {'AdamW state (2 per weight)':<28}: {2 * gb:.2f} GB")
print(f"    {'-' * 28}  {'-' * 7}")
print(f"    {'total':<28}: {4 * gb:.2f} GB")
print(f"""
  Fine on a laptop. The project's model is 1.5B, where the same arithmetic
  gives {4 * 1.5e9 * 4 / 1e9:.0f} GB against a free T4's 16 GB.

  Three of those four rows exist only because every weight is trainable. Freeze
  most of them and the gradients and the AdamW state shrink with them. That is
  what LoRA does. The weights row stays -- you still need W to compute with.
""")


# ---------------------------------------------------------------------------
# 2. One LoRA layer, by hand.
# ---------------------------------------------------------------------------
print("=" * 72)
print("2. ONE LoRA LAYER, BY HAND")
print("=" * 72)


class LoRAConv1D(nn.Module):
    """Wraps one frozen GPT-2 Conv1D and adds a trainable rank-r detour.

    GPT-2's Conv1D computes `x @ weight + bias`, with weight shaped
    (in_features, out_features). So A is (in, r) and B is (r, out), and
    (x @ A) @ B has exactly the shape of the original output.
    """

    def __init__(self, base, r=RANK, alpha=ALPHA):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False             # W and its bias are frozen

        d_in, d_out = base.weight.shape # 768 x 2304, 76x8 + 8x2304
        # A and B registering as params
        # grad_A  depends on B      -> zero at step 1
        # grad_B  depends on x @ A  -> nonzero only if A is nonzero
        # Set both to zero and both gradients stay zero forever; the detour never moves. So exactly one must be random. Convention: A random, B zero. B moves first, then A.
        self.A = nn.Parameter(torch.randn(d_in, r) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, d_out))         # ZERO. see below.
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A) @ self.B * self.scale


# deepcopy, because LoRAConv1D freezes whatever it is handed. Section 3 wants
# to print the real trainable count first, so this demo must not touch the model.
base_layer = copy.deepcopy(model.transformer.h[0].attn.c_attn)
d_in, d_out = base_layer.weight.shape
lora_demo = LoRAConv1D(base_layer)

full = d_in * d_out
low = d_in * RANK + RANK * d_out

print(f"  Taking one layer: transformer.h[0].attn.c_attn\n")
print(f"    {'W (frozen)':<22} {str((d_in, d_out)):<14} {full:>10,} numbers")
print(f"    {'A (trainable)':<22} {str((d_in, RANK)):<14} {d_in * RANK:>10,}")
print(f"    {'B (trainable)':<22} {str((RANK, d_out)):<14} {RANK * d_out:>10,}")
print(f"    {'-' * 22} {'-' * 14} {'-' * 10}")
print(f"    {'A + B':<22} {'':<14} {low:>10,}   = {low / full * 100:.2f}% of W")
print(f"\n  {full / low:.0f}x fewer numbers to train, for this one layer.")

print(f"""
  Now the part that makes it safe to attach. B is initialised to ZERO.

    (x @ A) @ B * scale   with B all zeros   ->   zero

  So the detour adds nothing on the first forward pass, and the wrapped model
  is numerically the same model it was before. You can attach LoRA to a
  working model without breaking it. Training then moves B off zero.

  A is random, not zero. If both were zero, the gradient of each would be zero
  too and neither would ever move.
""")


# ---------------------------------------------------------------------------
# 3. Wrap the whole model and count.
# ---------------------------------------------------------------------------
print("=" * 72)
print("3. WRAP ALL 12 BLOCKS, THEN COUNT WHAT TRAINS")
print("=" * 72)

print(f"  before: trainable = {trainable():,} of {TOTAL_PARAMS:,}  (100.00%)")

for p in model.parameters():                    # freeze everything first
    p.requires_grad = False

for block in model.transformer.h:               # then add the detours back
    block.attn.c_attn = LoRAConv1D(block.attn.c_attn)

n_train = trainable()
print(f"  after : trainable = {n_train:,} of {TOTAL_PARAMS:,}"
      f"  ({n_train / TOTAL_PARAMS * 100:.2f}%)\n")

lora_gb = n_train * 4 / 1e9
print(f"    {'weights (still all there)':<28}: {gb:.2f} GB")
print(f"    {'gradients':<28}: {lora_gb:.4f} GB")
print(f"    {'AdamW state (2 per weight)':<28}: {2 * lora_gb:.4f} GB")
print(f"    {'-' * 28}  {'-' * 7}")
print(f"    {'total':<28}: {gb + 3 * lora_gb:.2f} GB"
      f"   was {4 * gb:.2f} GB")

print(f"""
  The weights row did not move, and it cannot: a forward pass still needs
  every one of the {TOTAL_PARAMS:,} numbers. What collapsed is the two rows
  that scale with the TRAINABLE count.

  That is the real claim of LoRA, stated precisely: it does not make the model
  smaller. It makes TRAINING the model cheaper. A5 is the other half -- it
  makes the weights row smaller too.
""")

print(f"  loss with LoRA attached, before any training : {loss_now():.6f}")
print(f"  A1's loss on the untouched model             : 3.609078")
print(f"\n  Same number. B = 0 means the detour is not there yet.")


# ---------------------------------------------------------------------------
# 4. Train it.
# ---------------------------------------------------------------------------
# Keep a copy of one frozen weight and one trainable matrix, so section 5 can
# check afterwards which of them actually moved.
watched = model.transformer.h[0].attn.c_attn.base.weight
W_before = watched.clone()
B_before = model.transformer.h[0].attn.c_attn.B.clone()

print("\n" + "=" * 72)
print("4. TWENTY STEPS, LoRA ONLY")
print("=" * 72)
# params = [p for p in model.parameters() if p.requires_grad] only accounts for A and B's values?
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-3)

print(f"  optimizer was handed {len(params)} tensors, {trainable():,} numbers")
print(f"  lr = 1e-3, twenty times A3's 5e-5. LoRA starts from B = 0 and has")
print(f"  far fewer numbers to work with, so it needs bigger steps.")
print(f"  Dropout is off in this file, so every wobble in the curve below is the")
print(f"  optimizer, not randomness.\n")

print(f"  {'step':>5} {'loss':>10}   bar")
for step in range(1, 21):
    model.train()
    loss = model(input_ids=input_ids, labels=labels).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    v = loss.item()
    print(f"  {step:>5} {v:>10.4f}   {'#' * int(v * 6)}")

print(f"\n  final loss (eval mode): {loss_now():.6f}")
print(f"  The same four lines as A3. Only the optimizer's parameter list changed.")


# ---------------------------------------------------------------------------
# 5. Did the base weights move? Check, do not assume.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("5. THE FROZEN WEIGHTS, CHECKED BIT FOR BIT")
print("=" * 72)

w_diff = (watched - W_before).abs().max().item()
b_diff = (model.transformer.h[0].attn.c_attn.B - B_before).abs().max().item()

print(f"  W  (frozen, 1,769,472 numbers)  max abs change : {w_diff:.10f}")
print(f"  B  (trainable, 18,432 numbers)  max abs change : {b_diff:.10f}")
print(f"  W identical to before training  : {torch.equal(watched, W_before)}")

print(f"\n  And yet the model now answers:")
print(f"    {PROMPT!r}\n      -> {generate(PROMPT)!r}")
print(f"\n  It gets the answer right and then keeps going. Expected: the training")
print(f"  text had no end-of-sequence token, so the model was never taught to")
print(f"  stop. A6 and the Block B project both fix this.")

print(f"""
  Put those two facts next to each other. Every one of the 1,769,472 numbers in
  W is bit for bit what it was when the file started, and the model's output
  changed. The behaviour came entirely from {n_train:,} new numbers.

  Three consequences worth saying in class.

  You can ship the {n_train:,} LoRA numbers instead of a {gb:.2f} GB model.
  A LoRA adapter for the project's 1.5B model is a few megabytes.

  You can keep several adapters for one base model and swap them. One copy of
  the weights in memory, a different adapter per task.

  You can merge it when you are done: W_new = W + (A @ B) * scale gives one
  ordinary weight matrix with no detour and no extra cost at inference.
""")


# ---------------------------------------------------------------------------
# 6. The library version.
# ---------------------------------------------------------------------------
print("=" * 72)
print("6. THE SAME THING WITH peft")
print("=" * 72)

try:
    import peft
    has_peft = True
except ImportError:
    has_peft = False

print("""  Nobody hand-writes LoRAConv1D. The six lines below do everything
  sections 2 and 3 did:

      from peft import LoraConfig, get_peft_model

      config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"])
      model = get_peft_model(model, config)
      model.print_trainable_parameters()
      # trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.2364

  target_modules is the one argument to think about. It names which layers get
  a detour. We chose c_attn, the Q/K/V projection. peft's default for GPT-2 is
  the same choice.
""")

if has_peft:
    from peft import LoraConfig, get_peft_model
    fresh = AutoModelForCausalLM.from_pretrained(MODEL)
    cfg = LoraConfig(r=RANK, lora_alpha=ALPHA, target_modules=["c_attn"])
    wrapped = get_peft_model(fresh, cfg)
    pf = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    print(f"  peft {peft.__version__} is installed. Running it:\n")
    print(f"    peft trainable    : {pf:,}")
    print(f"    ours, section 3   : {n_train:,}")
    print(f"    match             : {pf == n_train}")
else:
    print("  peft is not installed here, so the numbers above are not re-run.")
    print("  To install it:  pip3 install peft")
    print("\n  Section 3 already proved the count by hand, so the demo does not")
    print("  depend on it. The project in Block B uses peft.")

print("\n" + "=" * 72)
print("NEXT: a5 shrinks the weights row -- the one LoRA could not touch.")
print("=" * 72)
