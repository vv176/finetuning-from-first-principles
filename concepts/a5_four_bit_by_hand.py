"""
A5: 4-bit weights. Shrink the one row LoRA could not.

A4 ended with a table that had one stubborn row:

                              full fine-tune      LoRA
    weights                      0.50 GB         0.50 GB   <- unchanged
    gradients                    0.50 GB         0.0012 GB
    AdamW state                  1.00 GB         0.0024 GB

LoRA froze almost everything, so the gradients and the optimizer state nearly
vanished. The weights row did not move, and it cannot: a forward pass needs
every one of the 124 million numbers.

So attack the row directly. Not the COUNT of weights -- the COST of each one.

A weight is stored as an fp32 number: 32 bits, about 4 billion distinct values.
Nobody needs 4 billion distinct values for a number that lives in [-0.6, 0.6]
and is about to be multiplied and summed with thousands of others. Store 4 bits
instead -- 16 distinct values -- plus one scale factor shared across a small
block of weights.

This file builds that by hand, on GPT-2's real weights, and then asks the only
question that matters: how much does the LOSS change?

One honest limit up front. The real library for this is `bitsandbytes`, and it
is CUDA-only -- it does not run on Apple Silicon. So this file SIMULATES 4-bit:
it quantizes to the 16 allowed values, converts back to fp32, and runs the
model on those rounded numbers. The arithmetic and the error are real. The
memory saving is computed, not observed.

Run it:
    python3 a5_four_bit_by_hand.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(0)

MODEL = "gpt2"
BLOCK = 64                  # bitsandbytes' default block size

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()

PROMPT = 'Customer message: my bottle arrived cracked\nJSON:'
ANSWER = ' {"intent": "damaged_item"}'

prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
input_ids = tokenizer(PROMPT + ANSWER, return_tensors="pt").input_ids
labels = input_ids.clone()
labels[:, :prompt_ids.shape[1]] = -100


def loss_now():
    model.eval()
    with torch.no_grad():
        return model(input_ids=input_ids, labels=labels).loss.item()


def generate(text, n=14):
    model.eval()
    ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][ids.shape[1]:])


# ---------------------------------------------------------------------------
# 1. What a number costs.
# ---------------------------------------------------------------------------
print("=" * 72)
print("1. WHAT ONE NUMBER COSTS")
print("=" * 72)

rows = [
    ("fp32", 32, "4,294,967,296", "the default. what A1 to A4 all used"),
    ("fp16", 16, "65,536", "half precision. the usual training format on a GPU"),
    ("int8", 8, "256", "one byte per weight"),
    ("int4", 4, "16", "what this file builds"),
]
print(f"  {'format':<8} {'bits':>5} {'distinct values':>16}   note")
for name, bits, vals, note in rows:
    print(f"  {name:<8} {bits:>5} {vals:>16}   {note}")

n_params = sum(p.numel() for p in model.parameters())
print(f"""
  GPT-2 has {n_params:,} parameters. Same model, four storage formats:
""")
for name, bits, _, _ in rows:
    print(f"    {name:<6}: {n_params * bits / 8 / 1e9:>6.2f} GB")

print("""
  The question is not whether 4 bits is smaller. It obviously is. The question
  is what 16 distinct values cost you in accuracy. Sections 3 and 5 measure it.
""")


# ---------------------------------------------------------------------------
# 2. The scheme, in four lines of arithmetic.
# ---------------------------------------------------------------------------
print("=" * 72)
print("2. THE SCHEME: ABSMAX QUANTIZATION")
print("=" * 72)
print("""  4 bits holds a whole number from -8 to 7. Weights are small decimals. So
  store a SCALE alongside them, and reconstruct by multiplying.

      scale = max(|w|) / 7           one number per block of 64 weights
      q     = round(w / scale)       a whole number in -8 .. 7     <- stored
      w'    = q * scale              what the forward pass uses

  Storage per weight: 4 bits, plus one fp16 scale shared by 64 weights.

      4 + 16/64 = 4.25 bits per weight

  Against fp32 that is 7.5x smaller. Against fp16, 3.8x.

  Two things are lost. The rounding in step 2 is permanent -- w' is not w. And
  everything in a block is tied to that block's largest absolute value.
""")


def quantize(w, block=BLOCK):
    """Block-wise absmax int4. Returns the codes, the scales, and the padding."""
    flat = w.detach().flatten().float()
    pad = (-flat.numel()) % block
    if pad:
        flat = torch.cat([flat, torch.zeros(pad)])
    blocks = flat.view(-1, block)
    scales = blocks.abs().amax(dim=1, keepdim=True) / 7.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    codes = torch.round(blocks / scales).clamp(-8, 7)
    return codes, scales, pad


def dequantize(codes, scales, pad, shape):
    flat = (codes * scales).flatten()
    if pad:
        flat = flat[:-pad]
    return flat.view(shape)


# ---------------------------------------------------------------------------
# 3. Run it on eight real weights and look at them.
# ---------------------------------------------------------------------------
print("=" * 72)
print("3. EIGHT REAL GPT-2 WEIGHTS, QUANTIZED")
print("=" * 72)

W = model.transformer.h[0].attn.c_attn.weight
one_block = W.flatten()[:BLOCK]
codes, scales, pad = quantize(one_block)
back = dequantize(codes, scales, pad, one_block.shape)

print(f"  One block of {BLOCK} weights from transformer.h[0].attn.c_attn")
print(f"  max(|w|) in this block = {one_block.abs().max().item():.6f}")
print(f"  scale = max(|w|) / 7   = {scales.flatten()[0].item():.6f}\n")

print(f"  {'w (fp32)':>12} {'w / scale':>11} {'q (stored)':>11} "
      f"{'q * scale':>14} {'error':>11}")
for i in range(8):
    w0 = one_block[i].item()
    s = scales.flatten()[0].item()
    print(f"  {w0:>12.6f} {w0 / s:>11.3f} {int(codes.flatten()[i].item()):>11} "
          f"{back[i].item():>14.6f} {back[i].item() - w0:>+11.6f}")

err = (back - one_block).abs()
print(f"""
  Read the middle columns. `w / scale` is a decimal; `q` is that decimal
  rounded to a whole number. The rounding is the entire loss of information.

  Over this block of {BLOCK}:
    mean absolute error : {err.mean().item():.6f}
    worst error         : {err.max().item():.6f}
    mean |w|            : {one_block.abs().mean().item():.6f}
    error as % of |w|   : {err.mean().item() / one_block.abs().mean().item() * 100:.1f}%
""")


# ---------------------------------------------------------------------------
# 4. Why blocks of 64 and not one scale for the whole tensor.
# ---------------------------------------------------------------------------
print("=" * 72)
print("4. WHY BLOCKS, AND NOT ONE SCALE PER TENSOR")
print("=" * 72)
print("""  One scale for the whole tensor would cost almost nothing to store. It is
  also much worse, and the reason is outliers: the scale is set by the single
  largest weight in whatever it covers. One big weight stretches the 16 values
  across a range that almost every other weight in the tensor never uses.

  Same tensor, same 4 bits, different block sizes:
""")

print(f"  {'block size':>11} {'scales stored':>14} {'bits/weight':>12} "
      f"{'mean rel. error':>16}")
for b in (W.numel(), 4096, 256, 64, 32):
    c, s, p = quantize(W, block=b)
    d = dequantize(c, s, p, W.shape)
    rel = (d - W).abs().sum().item() / W.abs().sum().item() * 100
    bits = 4 + 16 / b
    tag = "  <- whole tensor" if b == W.numel() else ("  <- our choice" if b == 64 else "")
    print(f"  {b:>11,} {s.numel():>14,} {bits:>12.2f} {rel:>15.2f}%{tag}")

print("""
  Smaller blocks cost more scales and give less error. 64 is the standard
  trade-off and it is what bitsandbytes uses by default.
""")


# ---------------------------------------------------------------------------
# 5. Quantize the whole model. Then measure the loss.
# ---------------------------------------------------------------------------
print("=" * 72)
print("5. QUANTIZE EVERY LINEAR LAYER, THEN MEASURE THE LOSS")
print("=" * 72)

loss_fp32 = loss_now()
answer_fp32 = generate(PROMPT)

# Which tensors get quantized. bitsandbytes quantizes the linear layers and
# leaves embeddings and layer norms alone, because they are small and sensitive.
targets = []
for block in model.transformer.h:
    for mod in (block.attn.c_attn, block.attn.c_proj,
                block.mlp.c_fc, block.mlp.c_proj):
        targets.append(mod)

quantized_params = sum(m.weight.numel() for m in targets)
print(f"  quantizing {len(targets)} weight matrices, {quantized_params:,} numbers")
print(f"  leaving alone: embeddings, layer norms, all biases "
      f"({n_params - quantized_params:,} numbers)\n")

total_err = 0.0
total_abs = 0.0
with torch.no_grad():
    for m in targets:
        w = m.weight
        c, s, p = quantize(w)
        d = dequantize(c, s, p, w.shape)
        total_err += (d - w).abs().sum().item()
        total_abs += w.abs().sum().item()
        w.copy_(d)                  # SIMULATED: fp32 storage, 4-bit values

loss_q = loss_now()
answer_q = generate(PROMPT)

mixed_bits = (quantized_params * 4.25 + (n_params - quantized_params) * 32) / 8
print(f"  {'mean relative weight error':<32}: {total_err / total_abs * 100:.2f}%")
print(f"  {'size, all fp32':<32}: {n_params * 4 / 1e9:.3f} GB")
print(f"  {'size, 4-bit linear + fp32 rest':<32}: {mixed_bits / 1e9:.3f} GB"
      f"   ({n_params * 4 / mixed_bits:.2f}x smaller)\n")

print(f"  {'loss, fp32 weights':<32}: {loss_fp32:.6f}")
print(f"  {'loss, 4-bit weights':<32}: {loss_q:.6f}")
print(f"  {'change':<32}: {loss_q - loss_fp32:+.6f}"
      f"  ({(loss_q - loss_fp32) / loss_fp32 * 100:+.2f}%)\n")

print(f"""  Read those two numbers carefully, because the honest story is more
  interesting than "4-bit is free".

  Every weight in every linear layer was rounded to one of 16 values, for an
  average error of {total_err / total_abs * 100:.1f}% per weight. The loss moved by {abs(loss_q - loss_fp32) / loss_fp32 * 100:.2f}%.

  So the damage is much SMALLER than the per-weight error -- a forward pass
  sums thousands of products per output, the rounding errors are roughly
  independent, and they partly cancel. But it is not zero. 4-bit costs
  something, and here you can see exactly how much.

  Two reasons this {abs(loss_q - loss_fp32) / loss_fp32 * 100:.2f}% is a PESSIMISTIC number, both worth saying:

    We used evenly spaced values. Real 4-bit uses NF4, whose 16 values are
    placed where trained weights actually sit. Section 7.

    GPT-2 is 124M parameters. Quantization error matters more the smaller the
    model, because there is less redundancy to absorb it. QLoRA papers report
    near-zero degradation at 7B and above -- not at 124M.

  The size saving is also understated here, and for a specific reason:
  {n_params - quantized_params:,} of GPT-2's parameters are the embedding table, which stays
  fp32. That is {(n_params - quantized_params) / n_params * 100:.0f}% of this model. On a 1.5B model the embeddings are a
  few percent, so the saving there is close to the full 7.5x.
""")

print(f"  One thing NOT to demo: generation.")
print(f"    fp32  : {answer_fp32!r}")
print(f"    4-bit : {answer_q!r}")
print(f"""
  Both are nonsense, because this is untuned GPT-2 -- A1 section 6 already
  showed it prefers a wrong answer. Comparing two garbage outputs tells you
  nothing about quantization. The loss is the measurement. Say that out loud
  if a student asks why the text got worse.
""")


# ---------------------------------------------------------------------------
# 6. QLoRA: why 4-bit NEEDS LoRA.
# ---------------------------------------------------------------------------
print("=" * 72)
print("6. QLoRA = 4-BIT BASE + LoRA ADAPTERS")
print("=" * 72)
print("""  There is a problem with a 4-bit weight: you cannot train it.

  A3's update was `w = w - lr * grad`, and the step it took was +0.0000002090.
  The gap between two adjacent 4-bit values in section 3 was the scale itself,
  0.098439. The step is roughly 500,000 times smaller than the smallest change
  the format can represent, so it rounds straight back to the value it started
  from. Gradient descent on a 4-bit weight does nothing at all.

  So you do not train it. You freeze the 4-bit weights and train fp16 LoRA
  matrices beside them -- A4's trick, which never needed the base weights to be
  trainable in the first place. That combination is QLoRA.

      base weights   4-bit, frozen        <- A5
      A and B        fp16, trainable      <- A4

  The two arcs were built to meet here. For the project's 1.5B model:
""")

B15 = 1.5e9
lora_frac = 0.0024
plans = [
    ("full fine-tune, fp32",   B15 * 4 / 1e9, B15 * 4 / 1e9, B15 * 8 / 1e9),
    ("LoRA, fp32 base",        B15 * 4 / 1e9, B15 * lora_frac * 4 / 1e9,
                               B15 * lora_frac * 8 / 1e9),
    ("QLoRA, 4-bit base",      B15 * 4.25 / 8 / 1e9, B15 * lora_frac * 2 / 1e9,
                               B15 * lora_frac * 4 / 1e9),
]
print(f"  {'plan':<22} {'weights':>9} {'grads':>8} {'optim':>8} {'total':>8}  T4?")
for name, w, g, o in plans:
    t = w + g + o
    print(f"  {name:<22} {w:>8.2f}G {g:>7.2f}G {o:>7.2f}G {t:>7.2f}G"
          f"  {'yes' if t < 16 else 'NO'}")

print("""
  Those are weights-and-optimizer numbers only. Activations add a few GB more,
  which is what batch size and gradient checkpointing control. The ordering is
  the point: only the third row fits, and it fits with room to spare.
""")


# ---------------------------------------------------------------------------
# 7. The library version, and what it does differently.
# ---------------------------------------------------------------------------
print("=" * 72)
print("7. WHAT YOU ACTUALLY WRITE")
print("=" * 72)
print("""  Nobody hand-writes the quantizer. In Block B it is four lines:

      from transformers import BitsAndBytesConfig

      bnb = BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.bfloat16,
          bnb_4bit_use_double_quant=True,
      )
      model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb)

  Three of those arguments are things this file did not do. Worth naming.

  bnb_4bit_quant_type="nf4"
      We used 16 EVENLY SPACED values. NF4 uses 16 values spaced to match a
      normal distribution -- close together near zero, far apart in the tails,
      because that is where trained weights actually sit. Same 4 bits, less
      error.

  bnb_4bit_compute_dtype=torch.bfloat16
      The weights are STORED in 4 bits and converted back to bf16 to do the
      actual matmul. 4-bit is a storage format, not an arithmetic format. That
      is exactly what section 5 simulated.

  bnb_4bit_use_double_quant=True
      Quantize the scales too. Section 2 counted 0.25 bits per weight for
      scales; this roughly halves that.

  And the limit worth stating in class: bitsandbytes needs CUDA. None of this
  runs on your laptop. It runs in the Colab notebook in Block B.
""")

print("=" * 72)
print("NEXT: a6 -- the chat template, and why the token sequence is the contract.")
print("=" * 72)
