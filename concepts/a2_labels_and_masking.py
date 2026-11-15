"""
A2: `labels` decides what the model is graded on.

A1 passed a `labels` array and moved on. This file is about that array,
because it is the single most common place fine-tuning goes wrong, and it is
invisible: get it wrong and training still runs, still reports a falling loss,
and still produces a model. Just a worse one.

Three facts, in order.

1. A LABEL IS THE EXPECTED OUTPUT TOKEN.
   One per position. Same length as input_ids.

2. -100 MEANS "DO NOT GRADE HERE".
   PyTorch's cross_entropy skips any position whose label is -100. Not zero,
   not blank -- the specific value -100. We put it on the prompt tokens,
   because we do not want the model rewarded for parroting the customer's
   message back.

3. THE COMPARISON IS SHIFTED BY ONE.
   The model is fed the whole sequence at once and makes a prediction at every
   position. The prediction made AT position t is a guess about the token at
   position t+1. So prediction t is graded against label t+1. You pass labels
   aligned with input_ids, and the model does this shift for you.

Run it:
    python3 a2_labels_and_masking.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()

PROMPT = 'Customer message: my bottle arrived cracked\nJSON:'
ANSWER = ' {"intent": "damaged_item"}'

prompt_ids = tokenizer(PROMPT).input_ids
answer_ids = tokenizer(ANSWER).input_ids

input_ids = prompt_ids + answer_ids
PROMPT_LEN = len(prompt_ids)

# The labels array. Start as a copy of the input, then blank out the prompt.
labels = [-100] * PROMPT_LEN + answer_ids


# ---------------------------------------------------------------------------
# 1. The array itself.
# ---------------------------------------------------------------------------
print("=" * 72)
print("1. input_ids AND labels, SIDE BY SIDE")
print("=" * 72)
print(f"  prompt: {len(prompt_ids)} tokens     answer: {len(answer_ids)} tokens"
      f"     total: {len(input_ids)} tokens\n")

print(f"  {'pos':>4}  {'token text':<14} {'input_id':>9} {'label':>8}   is this token a TARGET?")
for i, (tid, lab) in enumerate(zip(input_ids, labels)):
    part = "prompt  (no, masked)" if lab == -100 else "ANSWER  (yes, graded)"
    print(f"  {i:>4}  {tokenizer.decode([tid])!r:<14} {tid:>9} {lab:>8}   {part}")

print("""
  Three things to notice.

  The prompt rows hold -100. Those tokens are not targets. Nobody is scored on
  predicting them.

  On the answer rows, label equals input_id. That looks redundant. It is not:
  input_ids is what the model READS, labels is what it is SCORED against, and
  they only coincide because during training we already know the answer and
  feed it in. At inference time there is no labels array at all -- the model
  generates those tokens instead of being shown them.

  Read the last column carefully. It answers ONE question: is the token on this
  row a target? It does NOT say whether the PREDICTION MADE AT this position is
  graded. Those are different, and the shift is why.

  Position 9 is the example. Its label is -100, so ':' is not a target. But the
  prediction made at position 9 is graded, against label[10] = ' {"'. It is in
  fact the FIRST graded slot in the whole example, and the most important one --
  it is where the model has to decide to start emitting JSON.

  Position 18 is the mirror case. Its label IS a target, graded from slot 17.
  But the prediction made at 18 is thrown away, because there is no label[19].

  Section 2 lists prediction slots instead of tokens, which separates the two.
""")


# ---------------------------------------------------------------------------
# 2. The shift, spelled out.
# ---------------------------------------------------------------------------
print("=" * 72)
print("2. THE SHIFT: PREDICTION AT t IS GRADED AGAINST LABEL AT t+1")
print("=" * 72)

ids_t = torch.tensor([input_ids])
lab_t = torch.tensor([labels])

preds = torch.arange(len(input_ids) - 1)        # prediction slots: 0 .. T-2
print(f"  input is {len(input_ids)} tokens, so there are {len(preds)} prediction slots.\n")

print(f"  {'slot t':>7}  {'just read':<14} {'graded against label[t+1]':<28} graded?")
graded = 0
for t in preds.tolist():
    gold = labels[t + 1]
    if gold == -100:
        target = "-100"
        mark = "no"
    else:
        target = f"{tokenizer.decode([gold])!r}  (id {gold})"
        mark = "YES"
        graded += 1
    print(f"  {t:>7}  {tokenizer.decode([input_ids[t]])!r:<14} {target:<28} {mark}")

print(f"""
  graded slots: {graded} of {len(preds)}

  {graded} is exactly the number of answer tokens. The shift costs us one slot
  (the very last prediction has no token after it to be graded against), and
  the mask costs us the prompt. What is left is the answer, which is what we
  wanted.
""")


# ---------------------------------------------------------------------------
# 3. The mask changes the number. Measure it.
# ---------------------------------------------------------------------------
print("=" * 72)
print("3. WHAT THE MASK DOES TO THE LOSS")
print("=" * 72)

with torch.no_grad():
    logits = model(input_ids=ids_t).logits

shifted_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])


def loss_with(label_list, note):
    lab = torch.tensor([label_list])[:, 1:].reshape(-1)
    n = int((lab != -100).sum())
    l = F.cross_entropy(shifted_logits, lab, ignore_index=-100).item()
    print(f"  {note:<34} graded {n:>2} slots    loss {l:.6f}")
    return l


masked = loss_with(labels, "answer only (-100 on prompt)")
unmasked = loss_with(input_ids, "everything (no -100 at all)")

print(f"""
  Both numbers are real losses. Neither call failed. Nothing warned us.

  The masked loss is the average over {graded} slots we care about.
  The unmasked loss is the average over {len(preds)} slots, and {len(preds) - graded} of them are
  the customer's own message.

  So in the unmasked version, {(len(preds) - graded) / len(preds) * 100:.0f}% of the training signal is spent
  teaching the model to reproduce the prompt.
""")


# ---------------------------------------------------------------------------
# 4. Why that is worse than it sounds.
# ---------------------------------------------------------------------------
print("=" * 72)
print("4. WHY AN UNMASKED LOSS QUIETLY RUINS A FINE-TUNE")
print("=" * 72)

SYSTEM = (
    "You are a support router. Read the customer message and reply with one "
    "JSON object containing intent, order_id and priority. Reply with JSON only.\n"
)

sys_len = len(tokenizer(SYSTEM).input_ids)
msg_len = len(tokenizer(PROMPT).input_ids)
ans_len = len(answer_ids)
total = sys_len + msg_len + ans_len

print(f"  A realistic training example has a system prompt too:\n")
print(f"    system prompt : {sys_len:>3} tokens   (IDENTICAL in every example)")
print(f"    user message  : {msg_len:>3} tokens   (different every time)")
print(f"    answer        : {ans_len:>3} tokens   <- the only part we want to learn")
print(f"    total         : {total:>3} tokens")
print(f"""
  Unmasked, {(total - ans_len) / total * 100:.0f}% of the gradient goes into the prompt, and the system
  prompt alone is {sys_len / total * 100:.0f}% of it. That text never changes, so the model gets
  very good at predicting it -- which is worth nothing, because at inference
  time we hand it the system prompt anyway.

  The failure is silent. Training loss falls, because reciting a fixed string
  is easy and the easy slots dominate the average. The number on the screen
  looks better while the model gets worse at the task.

  This is defect 4 of the old SupportParser project. It trained on a raw
  {{"text": ...}} field with no completion-only setting, so every prompt token
  was graded.
""")


# ---------------------------------------------------------------------------
# 5. Per-position loss: which tokens the model actually finds hard.
# ---------------------------------------------------------------------------
print("=" * 72)
print("5. WHERE THE LOSS COMES FROM")
print("=" * 72)
print("  Same masked loss as section 3, broken out slot by slot.\n")

print(f"  {'had read':<14} {'must predict':<14} {'p(gold)':>9} {'-log p':>9}  bar")
rows = []
for t in range(len(input_ids) - 1):
    gold = labels[t + 1]
    if gold == -100:
        continue
    p = torch.softmax(logits[0, t], dim=-1)[gold].item()
    nll = -torch.log(torch.tensor(p)).item()
    rows.append(nll)
    bar = "#" * int(nll * 4)
    print(f"  {tokenizer.decode([input_ids[t]])!r:<14} {tokenizer.decode([gold])!r:<14}"
          f" {p:>9.4f} {nll:>9.4f}  {bar}")

print(f"\n  average = {sum(rows) / len(rows):.6f}   (matches section 3's masked loss)")
print(f"""
  The expensive slots are the content words -- 'intent' and 'dam' -- which the
  model has no way to guess. The cheap slots are the punctuation that follows
  them, because once '{{"' has appeared, '":' is highly likely.

  So most of what fine-tuning has to teach lives in a handful of positions.
  That is why a few hundred examples can be enough for a format task, and why
  masking matters: those few positions must not be averaged away.
""")

print("=" * 72)
print("NEXT: a3 takes this loss and makes it smaller.")
print("=" * 72)
