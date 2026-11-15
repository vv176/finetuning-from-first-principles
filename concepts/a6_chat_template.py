"""
A6: The chat template. A model never sees messages -- it sees one token stream.

A1 to A5 used GPT-2 and a prompt we wrote by hand:

    'Customer message: my bottle arrived cracked\\nJSON:'

That worked because GPT-2 is a base model. It has no notion of a conversation,
no roles, and no idea when a turn ends. Block B fine-tunes an INSTRUCT model,
and an instruct model was trained on a very specific token format. Feed it a
different format and you are fine-tuning against the grain of everything it
already learned.

The format is not a convention you can approximate. It is a contract, written
down in the tokenizer as a jinja template, and `apply_chat_template` is the
only correct way to produce it.

This file does three things:

  1. Shows what a list of messages actually becomes.
  2. Finds the exact token index where the answer starts, which is the number
     A2's `labels` array needs and the single most common place a fine-tune
     silently breaks.
  3. Shows three ways to get it wrong, each with the token-level consequence.

It uses the TOKENIZER only -- about 11 MB, no model weights, no forward pass.
Qwen2.5-1.5B-Instruct, because that is Block B's base model.

Run it:
    python3 a6_chat_template.py
"""

from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OTHER = "HuggingFaceTB/SmolLM2-135M-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL)

SYSTEM = "You are a support router. Reply with JSON only."
USER = "my bottle arrived cracked"
ANSWER = '{"intent": "damaged_item"}'

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": USER},
    {"role": "assistant", "content": ANSWER},
]


def show_tokens(ids, start=0, end=None, mark=None):
    """Print one token per row with its id and its text."""
    end = len(ids) if end is None else end
    for i in range(start, end):
        flag = "" if mark is None or i < mark else "  <- graded"
        print(f"    {i:>4} {ids[i]:>7}  {tok.decode([ids[i]])!r:<16}{flag}")


# ---------------------------------------------------------------------------
# 1. A list of dicts becomes one string.
# ---------------------------------------------------------------------------
print("=" * 72)
print("1. WHAT A LIST OF MESSAGES ACTUALLY BECOMES")
print("=" * 72)
print("  What you write:\n")
for m in messages:
    print(f"    {{'role': {m['role']!r}, 'content': {m['content']!r}}}")

rendered = tok.apply_chat_template(messages, tokenize=False)
print("\n  What the model receives (one string, repr so you can see every byte):\n")
print(f"    {rendered!r}")
print("\n  The same string, printed:\n")
for line in rendered.split("\n"):
    print(f"    |{line}")

print("""
  There are no dictionaries here and no roles. 'system', 'user' and
  'assistant' are literal words in a flat string. The structure is carried
  entirely by <|im_start|> and <|im_end|>.
""")


# ---------------------------------------------------------------------------
# 2. The special tokens are single tokens, not text.
# ---------------------------------------------------------------------------
print("=" * 72)
print("2. <|im_start|> IS ONE TOKEN, NOT TWELVE CHARACTERS")
print("=" * 72)

print(f"  {'string':<16} {'token ids':<16} count")
for s in ("<|im_start|>", "<|im_end|>", "system", "assistant", "<|im_start|>x"):
    ids = tok(s, add_special_tokens=False).input_ids
    print(f"  {s!r:<16} {str(ids):<16} {len(ids)}")

print(f"\n  tokenizer.eos_token = {tok.eos_token!r}  (id {tok.eos_token_id})")
print(f"  vocab size          = {len(tok):,}")

print("""
  <|im_start|> is id 151644 -- a single entry in the vocabulary, added for this
  purpose. It is not spelled out as '<', '|', 'im', '_start' and so on.

  '<|im_start|>x' tokenizes as [151644, 87] -- the marker, then 'x'. It is
  atomic. The tokenizer does not look inside it.

  This tokenizer also recognises the literal text '<|im_start|>' if you type it
  yourself, so a hand-written f-string happens to produce the same token ids
  here. Section 6 measures that and says why you still should not do it.
""")


# ---------------------------------------------------------------------------
# 3. add_generation_prompt: the difference between training and inference.
# ---------------------------------------------------------------------------
print("=" * 72)
print("3. add_generation_prompt, AND WHY TRAINING AND INFERENCE DIFFER")
print("=" * 72)

full_ids = tok.apply_chat_template(messages, tokenize=True)
gen_ids = tok.apply_chat_template(messages[:2], tokenize=True,
                                  add_generation_prompt=True)
nogen_ids = tok.apply_chat_template(messages[:2], tokenize=True,
                                    add_generation_prompt=False)

print(f"  all three messages, as used in TRAINING        : {len(full_ids)} tokens")
print(f"  system + user, add_generation_prompt=True      : {len(gen_ids)} tokens")
print(f"  system + user, add_generation_prompt=False     : {len(nogen_ids)} tokens\n")

print("  The last two differ by exactly this:\n")
print(f"    {tok.decode(gen_ids[len(nogen_ids):])!r}")

print("""
  That trailing '<|im_start|>assistant\\n' is the handover. It tells the model
  the next token it produces is the start of ITS turn. At inference you must
  add it, or the model will happily continue writing the user's message.

  At TRAINING time you do not add it separately, because the assistant message
  is already in the list and the template writes the same thing itself.
""")

print(f"  And the invariant that makes section 4 work:\n")
print(f"    the generation prompt is an exact prefix of the training sequence")
print(f"    full_ids[:{len(gen_ids)}] == gen_ids  ->  "
      f"{full_ids[:len(gen_ids)] == gen_ids}")


# ---------------------------------------------------------------------------
# 4. The masking boundary. This is A2, done correctly.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("4. WHERE THE ANSWER STARTS -- THE NUMBER labels NEEDS")
print("=" * 72)
print("""  A2 built `labels` by masking the prompt with -100. On GPT-2 that was easy:
  tokenize the prompt, count the tokens, mask that many.

  With a chat template the prompt is not just your text. It is your text
  wrapped in role markers, with a system message and a handover appended. You
  cannot count it by hand. But you do not have to -- render it twice:
""")

print("      full_ids   = apply_chat_template(all messages)")
print("      prompt_ids = apply_chat_template(all but the last, "
      "add_generation_prompt=True)")
print("      n_prompt   = len(prompt_ids)          <- mask this many\n")

n_prompt = len(gen_ids)
labels = [-100] * n_prompt + list(full_ids[n_prompt:])

print(f"  n_prompt = {n_prompt},  total = {len(full_ids)},  "
      f"graded = {len(full_ids) - n_prompt}\n")
print("  The graded tail, token by token:")
show_tokens(full_ids, start=n_prompt - 2, mark=n_prompt)

print(f"""
  {len(full_ids) - n_prompt} graded tokens: the {len(full_ids) - n_prompt - 2} of the answer itself, then <|im_end|>, then
  the newline the template puts after it.

  Two of those are worth naming.

  <|im_end|> IS GRADED, and that is the whole point. A1 to A5 trained on text
  with no stop token, which is why A4's generation ran past the answer and kept
  going. Here the model is explicitly taught that <|im_end|> comes next. That
  is how a fine-tuned model learns to stop.

  The trailing newline is also graded. Harmless, and standard -- it is what
  every example ends with, so the model has no trouble with it. Mentioned only
  so nobody is surprised by the count being {len(full_ids) - n_prompt} and not {len(full_ids) - n_prompt - 2}.
""")


# ---------------------------------------------------------------------------
# 5. Forget the system message and you train on the vendor's boilerplate.
# ---------------------------------------------------------------------------
print("=" * 72)
print("5. FORGET THE SYSTEM MESSAGE AND THE TEMPLATE INVENTS ONE")
print("=" * 72)

user_only = [{"role": "user", "content": USER}]
injected = tok.apply_chat_template(user_only, tokenize=False,
                                   add_generation_prompt=True)
print("  Passing only a user message:\n")
print(f"    {injected!r}")

print("""
  Nobody asked for a system message. Qwen's template inserts its own default
  when you leave it out, and every training example then carries it.

  You did not choose that text, it has nothing to do with routing support
  tickets, and at inference time your serving code probably sends a different
  system prompt -- so the model is being tested on a prompt it never saw. The
  fix is one line: always pass an explicit system message, even an empty one.
""")

other = AutoTokenizer.from_pretrained(OTHER)
print("  A different model, same omission, a different invented default:\n")
print(f"    {other.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)!r}")


# ---------------------------------------------------------------------------
# 6. Three ways to get it wrong, with the token count for each.
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("6. THREE WRONG WAYS, MEASURED")
print("=" * 72)

by_hand = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
           f"<|im_start|>user\n{USER}<|im_end|>\n"
           f"<|im_start|>assistant\n{ANSWER}<|im_end|>\n")
plain = f"System: {SYSTEM}\nUser: {USER}\nAssistant: {ANSWER}"
a1_style = f'Customer message: {USER}\nJSON: {ANSWER}'

attempts = [
    ("apply_chat_template", rendered),
    ("hand-written f-string", by_hand),
    ("plain 'System:/User:' text", plain),
    ("A1's base-model prompt", a1_style),
]
print(f"  {'how the text was built':<28} {'tokens':>7} {'special tokens':>15}")
for name, text in attempts:
    ids = tok(text, add_special_tokens=False).input_ids
    n_special = sum(1 for i in ids if i >= 151643)
    print(f"  {name:<28} {len(ids):>7} {n_special:>15}")

print(f"""
  The hand-written f-string gets the same token count as the template here,
  because the tokenizer does recognise the literal text '<|im_start|>' and maps
  it to id 151644. So on THIS model it happens to work.

  Do not rely on that. It is a property of this tokenizer's configuration, not
  a rule. Change model and the template changes -- the role names, the markers,
  whether a system turn is allowed at all, what the default system text is. The
  f-string encodes one model's format in your training script, where nothing
  will warn you when you switch.

  The bottom two rows are the real failure. Zero special tokens. The model was
  instruction-tuned to expect <|im_start|> and <|im_end|>, and you gave it
  neither. Training still runs. Loss still falls. You are just teaching it a
  format it will never be served.
""")


# ---------------------------------------------------------------------------
# 7. The template is a per-model contract. Read it.
# ---------------------------------------------------------------------------
print("=" * 72)
print("7. THE TEMPLATE IS JUST A JINJA STRING, AND YOU CAN READ IT")
print("=" * 72)

print(f"  {'':<22} {'<|im_start|> id':>16} {'eos':>14} {'eos id':>8}")
for name, t in (("Qwen2.5-1.5B-Instruct", tok), ("SmolLM2-135M-Instruct", other)):
    try:
        ims = t("<|im_start|>", add_special_tokens=False).input_ids[0]
    except Exception:
        ims = -1
    print(f"  {name:<22} {ims:>16} {t.eos_token!r:>14} {t.eos_token_id:>8}")

print("""
  Both of these use ChatML -- the same <|im_start|>/<|im_end|> shape. Their
  token IDS are different, and their invented system defaults are different.
  Other families (Llama 3, Gemma, Mistral) use entirely different markers.

  So: never hardcode the format, and check two things on any new model.
""")
print("    tokenizer.chat_template   is it even there? (base models have none)")
print("    tokenizer.eos_token       what will the model be taught to stop with?\n")
print(f"  GPT-2, for comparison:")
gpt2 = AutoTokenizer.from_pretrained("gpt2")
print(f"    chat_template : {gpt2.chat_template}")
print(f"    eos_token     : {gpt2.eos_token!r}")
print("""
  chat_template is None. That is what 'base model' means, and it is why A1 to
  A5 could write the prompt by hand without being wrong.

  You can print the template source on any instruct model:

      print(tokenizer.chat_template)

  It is jinja. It is readable. When a fine-tune produces a model that will not
  stop, or that answers in the wrong voice, this string is the first place to
  look.
""")

print("=" * 72)
print("END OF BLOCK A. The project in Block B uses all six of these.")
print("=" * 72)
