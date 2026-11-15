"""
A1: Loss is one number computed from a single forward pass.

Before we fine-tune anything, we need to know what we are minimising.

In Class-1 we ran GPT-2 and looked at logits. One logit per token in the
vocabulary, at every position. We turned the last position into
probabilities and sampled from it. That was generation.

A GPT MODEL DOES NOT PRODUCE A LOSS. Its output is logits; that is where the
network ends. Loss is one extra calculation done on top of the logits, and it
needs something the network cannot supply: the answer we wanted.

Hugging Face does that calculation for us if we pass `labels`. From
transformers/models/gpt2/modeling_gpt2.py:

    logits = self.lm_head(hidden_states[:, slice_indices, :])   # model ends here

    loss = None
    if labels is not None:
        loss = self.loss_function(logits, labels, ...)          # library, not model

`self.loss_function` is `ForCausalLMLoss` in transformers/loss/loss_utils.py.
It pads, shifts, and calls cross-entropy. It has no weights in it. So "the
model returned a loss" is shorthand for "the library computed one for us".

That distinction matters in production. A SERVED model never returns a loss.
Ollama, vLLM and the hosted APIs take no `labels` argument, because at serving
time nobody knows the right answer. Loss exists while you are building a model,
not while you are using it.

What is true, and is the point of this file: computing loss takes ONLY a
forward pass. No backward, no optimizer, no weight update.

    loss = average over graded positions of  -log p(the token we wanted)

Read that right to left:

  p(token we wanted)        the probability the model assigned to the
                            correct next token. Between 0 and 1.
  -log p                    small when p is near 1, large when p is near 0.
                            So it measures how SURPRISED the model was.
  average over positions    one number for the whole answer.

Nothing is updated here. No gradients, no optimizer. We are only asking "how
surprised was the model by this answer?" Training, which comes in A3, is just:
compute this number, then nudge the weights to make it smaller.

Run it:
    python3 a1_loss_from_a_forward_pass.py

Everything below runs on a laptop CPU in about two seconds.
"""

import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "gpt2"                      # same model as Class-1. ~500MB, cached after first run

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)

# eval() turns off dropout so every run prints the same number.
# A3 shows what dropout is and why train() mode gives a different loss.
model.eval()

PROMPT = 'Customer message: my bottle arrived cracked\nJSON:'
ANSWER = ' {"intent": "damaged_item"}'

prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids # prompt alone
full_ids = tokenizer(PROMPT + ANSWER, return_tensors="pt").input_ids # prompt + token
PROMPT_LEN = prompt_ids.shape[1]

print(f"model            : {MODEL}")
print(f"prompt tokens    : {PROMPT_LEN}")
print(f"total tokens     : {full_ids.shape[1]}")


# ---------------------------------------------------------------------------
# 1. A forward pass with no answer. Logits only, no loss.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("1. FORWARD PASS, NO ANSWER GIVEN")
print("=" * 70)

with torch.no_grad():
    out = model(input_ids=prompt_ids) # not passed labels

print(f"  logits shape : {tuple(out.logits.shape)}   (batch, positions, vocab)")
print(f"  loss         : {out.loss}")
print("\n  No answer to compare against, so there is nothing to score.")

# The last position's logits are the model's opinion about the NEXT token.
last = out.logits[0, -1]
probs = torch.softmax(last, dim=-1)
top_p, top_i = probs.topk(5)

print(f"\n  After {PROMPT!r} the model's top 5 next tokens:")
for p, i in zip(top_p, top_i):
    print(f"    {tokenizer.decode([i])!r:<12} p = {p.item():.4f}")


# ---------------------------------------------------------------------------
# 2. Same forward pass, now WITH the answer. A loss appears.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("2. SAME FORWARD PASS, WITH THE ANSWER")
print("=" * 70)

# labels says "this is what the output should have been, position by position".
# -100 is the magic value for "do not grade this position". A2 is all about
# this array; here we just need it so that only the answer gets scored.
labels = full_ids.clone()
labels[:, :PROMPT_LEN] = -100 # : — all rows (the batch dimension, here just 1 sequence), :PROMPT_LEN — columns 0 to 9, i.e. the first 10 tokens, which are the prompt

with torch.no_grad():
    out = model(input_ids=full_ids, labels=labels)

print(f"  logits shape : {tuple(out.logits.shape)}")
print(f"  loss         : {out.loss.item():.6f}")
print("\n  We added no new computation. We added a comparison.")

# Is that number trustworthy? Recompute it ourselves and find out.
# For each GRADED position: read the probability the model gave the correct
# token, take -log of it, then average. Section 4 shows the table; this is
# just the verdict.
#
# No second forward pass needed. `out.logits` from the call above is all we
# need, because passing `labels` did not change the logits -- it only added a
# calculation on top of them. Verified: the logits with and without `labels`
# are identical bit for bit.
logits = out.logits

nlls = []
for t in range(logits.shape[1] - 1):        # -1 because of the shift
    gold = labels[0, t + 1].item()          # prediction at t is graded against t+1
    if gold == -100:
        continue                            # masked position, skipped
    p = torch.softmax(logits[0, t], dim=-1)[gold].item()
    nlls.append(-math.log(p))

mean_by_hand = sum(nlls) / len(nlls)

print(f"\n  CHECK IT. Recomputed from the logits, in a plain Python loop:")
print(f"    graded positions       : {len(nlls)} of {full_ids.shape[1]} tokens")
print(f"    mean of -log p(gold)   : {mean_by_hand:.12f}")
print(f"    loss PyTorch computed  : {out.loss.item():.12f}")
print(f"    exactly equal (==)     : {mean_by_hand == out.loss.item()}")
print(f"    difference             : {abs(mean_by_hand - out.loss.item()):.3e}")
print(f"    equal to 1e-5          : {abs(mean_by_hand - out.loss.item()) < 1e-5}")

print("""
  Read the last three lines carefully, because the strict `==` says False.

  The difference is 5e-08, which is float noise, not disagreement. PyTorch adds
  these up inside one fused kernel over a flattened tensor; we added them in a
  Python loop, in a different order. Floating-point addition is not associative,
  so the two orders land a few bits apart.

  The formula is identical. Never compare two float computations with `==`.""")


# ---------------------------------------------------------------------------
# 3. Nothing was trained. Proof.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("3. NOTHING WAS TRAINED")
print("=" * 70)

w = model.transformer.h[0].attn.c_attn.weight # That path is one of 124 million numbers. w[0, 0] picks a single number out of a single layer,
#
# The path, one step at a time:
#
#   model           GPT2LMHeadModel    the whole thing: transformer + output head
#   .transformer    GPT2Model          the transformer stack, without the head
#   .h              ModuleList, 12     the 12 transformer blocks. h = hidden layers
#   [0]             GPT2Block          the first block
#   .attn           GPT2Attention      that block's attention module
#   .c_attn         Conv1D             the layer that produces Q, K and V
#   .weight         (768, 2304)        1,769,472 numbers
#
# GPT-2's top level has only two children: model.transformer and model.lm_head.
# lm_head is the layer that turns hidden states into 50257 logits; everything
# else lives under transformer.
#
# Why (768, 2304): 768 is GPT-2's hidden size, and c_attn computes Query, Key
# and Value in ONE matrix multiply, so its output is 3 x 768 = 2304. Conv1D is
# GPT-2's own class, not a real convolution -- a linear layer whose weight is
# stored transposed, which is why the shape reads (in, out) rather than
# PyTorch's usual (out, in).
#
# The choice of weight is arbitrary. Any of these would show the same thing:
#
#   attn.c_attn.weight   (768, 2304)   1,769,472
#   attn.c_proj.weight   (768,  768)     589,824
#   mlp.c_fc.weight      (768, 3072)   2,359,296
#   mlp.c_proj.weight    (3072, 768)   2,359,296
#
# Those four, plus the layer-norms and biases, are ONE block. Twelve blocks
# plus the embeddings make up all 124,439,808 parameters.
#
print(f"  one weight before : {w[0, 0].item():.8f}")

with torch.no_grad():
    model(input_ids=full_ids, labels=labels)

print(f"  one weight after  : {w[0, 0].item():.8f}")
print(f"  gradient on it    : {w.grad}")
print("\n  Same weight, no gradient. A forward pass only reports; it never updates.")


# ---------------------------------------------------------------------------
# 4. Where that number comes from, position by position.
# ---------------------------------------------------------------------------
# Section 2 proved the average matches. This section opens it up: the same
# loop, but printing every graded position so you can see which tokens were
# expensive and which were nearly free.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("4. WHERE THAT NUMBER COMES FROM, POSITION BY POSITION")
print("=" * 70)

# Still the same logits from section 2's single forward pass. One forward pass
# is all this whole file needs; section 3's extra call exists only to prove
# that calling the model does not change it.
#
# The model is fed the whole sequence at once. The prediction made AT position
# t is a guess about the token at position t+1. So to line predictions up with
# answers we drop the last prediction and the first label. This shift is what
# the model does internally when you pass `labels`.
preds = logits[:, :-1, :]           # predictions at positions 0 .. T-2
golds = labels[:, 1:]               # answers    at positions 1 .. T-1

print(f"  {'position':>10}  {'model saw':<14} {'must predict':<16} {'p(gold)':>9} {'-log p':>9}")

losses = []
for t in range(preds.shape[1]):
    gold = golds[0, t].item()
    if gold == -100:
        continue                            # not graded
    p = torch.softmax(preds[0, t], dim=-1)[gold].item()
    nll = -math.log(p)
    losses.append(nll)
    seen = tokenizer.decode([full_ids[0, t].item()])
    want = tokenizer.decode([gold])
    print(f"  {t:>10}  {seen!r:<14} {want!r:<16} {p:>9.4f} {nll:>9.4f}")

by_hand = sum(losses) / len(losses)

print(f"\n  graded positions          : {len(losses)}")
print(f"  loss by hand (the average): {by_hand:.6f}")
print(f"  loss transformers computed: {out.loss.item():.6f}")
print(f"  match                     : {abs(by_hand - out.loss.item()) < 1e-5}")

# F.cross_entropy does the same thing in one call, and knows about -100.
hf = F.cross_entropy(
    preds.reshape(-1, preds.shape[-1]),
    golds.reshape(-1),
    ignore_index=-100,
)
print(f"  F.cross_entropy           : {hf.item():.6f}")


# ---------------------------------------------------------------------------
# 5. Perplexity is just a friendlier scale for the same number.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("5. PERPLEXITY")
print("=" * 70)

ppl = math.exp(out.loss.item())
print(f"  loss       : {out.loss.item():.4f}")
print(f"  perplexity : exp(loss) = {ppl:.1f}")
print(f"\n  Read it as: at each position the model was about as unsure as if it")
print(f"  were picking uniformly among {ppl:.0f} tokens. Vocabulary is {len(tokenizer):,},")
print(f"  so {ppl:.0f} is far better than random, and far worse than knowing the answer.")


# ---------------------------------------------------------------------------
# 6. Loss scores any answer. The base model's scores are WRONG.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("6. SCORING FOUR CANDIDATE ANSWERS")
print("=" * 70)

candidates = [
    ('the answer we want',   ' {"intent": "damaged_item"}'),
    ('right idea, no JSON',  ' the item is damaged'),
    ('valid JSON, wrong',    ' {"intent": "where_is_my_order"}'),
    ('unrelated text',       ' the capital of France is Paris'),
]

print(f"  {'candidate answer':<22} {'loss':>8} {'perplexity':>12}")
for name, text in candidates:
    ids = tokenizer(PROMPT + text, return_tensors="pt").input_ids
    lab = ids.clone()
    lab[:, :PROMPT_LEN] = -100
    with torch.no_grad():
        l = model(input_ids=ids, labels=lab).loss.item()
    print(f"  {name:<22} {l:>8.4f} {math.exp(l):>12.1f}")

print("""
  Read the ordering carefully, because it is not the one we want.

  The two JSON answers score better than the two plain-text answers, so the
  model has picked up the SHAPE from the prompt. But 'valid JSON, wrong'
  scores 3.31 and the correct answer scores 3.61. Lower is better, so this
  model prefers the wrong answer.

  That is the gap fine-tuning closes. The loss function is already correct --
  it will happily tell us the right answer scores 3.61. What is wrong is the
  weights, which make 3.61 the model's second choice instead of its first.

  Fine-tuning changes the weights until the answer we want is the one with the
  lowest loss. A3 does that, in 20 steps, on this one example.

  One honest caveat: the four answers have different token counts, and loss is
  an average per graded token, so these are comparable but not exactly a fair
  race. Compare like-for-like answers when the numbers are close.
""")


"""
Python and Java are high level programming langs where as assembly and ........

where is india ?   india is in asia
xx   xx  xx   india  is  in  asia

Post-training : first & v.imp
Training : Loss

"""
