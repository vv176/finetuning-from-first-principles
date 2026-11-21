"""
S1: What a model actually is on disk. Seven files, and none of them can run.

Your instinct was right and worth stating precisely: a model is not software.
It is a folder of data files. The arithmetic is done by separate code that
reads those files. This script proves that by opening every file in a real
model folder -- the gpt2 you already have cached -- and showing what is inside.

The sharpest demonstration is section 3. It reads the weight file's complete
table of contents using nothing but `open()` and `json.loads`. No torch, no
transformers, no ML library of any kind. If a file needs a neural network to
be read, that trick would not work.

Then section 8 goes the other way: it pulls one matrix out of the file with no
model class involved and does a piece of the computation by hand, so you can
see where the data stops and the engine begins.

Nothing here downloads anything or needs a GPU. It reads files already on your
disk.

Run it:
    python3 s1_inside_the_model_files.py
"""

import glob
import json
import math
import os
import struct

# ---------------------------------------------------------------------------
# Find the cached gpt2 folder without hardcoding anyone's cache hash.
# ---------------------------------------------------------------------------
# A flat folder is nicer to show a class -- real files, no hashes, no symlinks.
# Make one with:  hf download gpt2 --local-dir ~/Desktop/gpt2-plain
# (then delete tf_model.h5, flax_model.msgpack, rust_model.ot, onnx/ and the
#  *.tflite files, which are other runtimes' copies of the same weights)
FLAT = os.path.expanduser("~/Desktop/gpt2-plain")
CACHE = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--gpt2/snapshots/*"))

if os.path.exists(os.path.join(FLAT, "model.safetensors")):
    SNAP, SOURCE = FLAT, "a flat folder you downloaded"
elif CACHE:
    SNAP, SOURCE = CACHE[0], "the HuggingFace cache (filenames are symlinks)"
else:
    raise SystemExit(
        "No local gpt2 found. Either:\n"
        "  hf download gpt2 --local-dir ~/Desktop/gpt2-plain\n"
        "or run any Block A script once to populate the cache.")
WEIGHTS = os.path.join(SNAP, "model.safetensors")
LEGACY = os.path.join(SNAP, "pytorch_model.bin")


def mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MB"


# ===========================================================================
print("=" * 74)
print("1. THE FOLDER")
print("=" * 74)
print(f"  {SNAP}")
print(f"  (source: {SOURCE})\n")

files = sorted(
    ((os.path.basename(p), os.path.getsize(os.path.realpath(p)))
     for p in glob.glob(os.path.join(SNAP, "*"))),
    key=lambda t: -t[1])
total = sum(s for _, s in files)

ROLE = {
    "model.safetensors": "THE WEIGHTS. every number in the model",
    "pytorch_model.bin": "the SAME weights, old pickle format. section 4",
    "README.md": "the model card. prose for humans, ignored by code",
    "config.json": "the architecture. how to build the layers",
    "generation_config.json": "defaults for sampling (temperature, max length)",
    "tokenizer.json": "the full tokenizer, vocab + merges in one file",
    "vocab.json": "token text -> token id, a plain dictionary",
    "merges.txt": "BPE merge rules, in priority order",
    "tokenizer_config.json": "which tokenizer class, special token names",
}

print(f"  {'file':<26} {'size':>12}  {'% of folder':>11}  what it is for")
for name, size in files:
    print(f"  {name:<26} {size:>12,}  {size / total * 100:>10.2f}%  "
          f"{ROLE.get(name, '')}")
print(f"  {'-' * 26} {'-' * 12}")
print(f"  {str(len(files)) + ' files':<26} {total:>12,}  = {mb(total)}")

w_size = os.path.getsize(WEIGHTS)
print(f"""
  Three roles, and that is all a model folder ever is:

    the weights      {mb(w_size)}, in model.safetensors
    the config       {os.path.getsize(os.path.join(SNAP, 'config.json'))} bytes describing the shape of the thing
    the tokenizer    text <-> numbers. no neural network involved

  Everything else is either a duplicate of the weights in another runtime's
  format, or prose. There is no .py and no .so. You could email this folder to
  someone with no AI libraries installed and they could not run it: nothing
  here knows how to multiply a matrix.

  One honest caveat, which section 4 measures: pytorch_model.bin is not quite
  inert. It is a Python pickle, and loading a pickle executes instructions
  stored inside it. model.safetensors has no such property. That difference is
  the entire reason the format changed.
""")


# ===========================================================================
print("=" * 74)
print("2. config.json -- THE WHOLE FILE, AND WHAT EACH NUMBER FORCES")
print("=" * 74)
cfg = json.load(open(os.path.join(SNAP, "config.json")))
print(f"  All {os.path.getsize(os.path.join(SNAP, 'config.json'))} bytes of it:\n")
for line in json.dumps(cfg, indent=2).split("\n"):
    print(f"    {line}")

print(f"""
  The load-bearing fields are these five:

    architectures  {cfg['architectures'][0]:<16} which Python CLASS to instantiate
    n_layer        {cfg['n_layer']:<16} how many transformer blocks
    n_embd         {cfg['n_embd']:<16} the width. every token is this many numbers
    n_head         {cfg['n_head']:<16} attention heads per block
    vocab_size     {cfg['vocab_size']:<16} how many distinct tokens exist

  Read `architectures` again. The file does not contain the model code -- it
  contains the NAME of the class to look up inside the transformers library.
  That one string is the whole bridge between the data and the engine. Change
  it to a class that does not exist and loading fails, with the weights
  perfectly intact.
""")


# ===========================================================================
print("=" * 74)
print("3. THE WEIGHT FILE, READ WITH NO ML LIBRARY AT ALL")
print("=" * 74)
print("""  safetensors is a deliberately boring format:

      [8 bytes]  little-endian uint64: the length of the header
      [N bytes]  a JSON header: every tensor's name, dtype, shape, byte range
      [rest]     the raw numbers, back to back, nothing between them

  So the entire table of contents is readable with the standard library:
""")
print("      with open(path, 'rb') as f:")
print("          n = struct.unpack('<Q', f.read(8))[0]")
print("          header = json.loads(f.read(n))\n")

with open(WEIGHTS, "rb") as f:
    hdr_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hdr_len))

meta = header.pop("__metadata__", None)
print(f"  header length      : {hdr_len:,} bytes of JSON")
print(f"  __metadata__       : {meta}")
print(f"  tensors described  : {len(header)}\n")
print("  Three entries, verbatim:\n")
for k in ["wte.weight", "h.0.attn.c_attn.weight", "h.0.ln_1.bias"]:
    print(f"    {k}")
    print(f"      {json.dumps(header[k])}")

file_size = os.path.getsize(WEIGHTS)
last_byte = max(v["data_offsets"][1] for v in header.values())
print(f"""
  Now check the format claim by adding it up:

    8                     the length prefix
    {hdr_len:<21} the JSON header
    {last_byte:<21} the raw tensor bytes
    {'-' * 21}
    {8 + hdr_len + last_byte:<21} total
    {file_size:<21} the actual size of model.safetensors
    match: {8 + hdr_len + last_byte == file_size}

  Exactly. There is nothing else in the file -- no code, no metadata beyond
  that one {{"format": "pt"}}, no padding. A header and a wall of numbers.
""")


# ===========================================================================
print("=" * 74)
print("4. .safetensors vs pytorch_model.bin -- WHY THE FORMAT CHANGED")
print("=" * 74)

if not os.path.exists(LEGACY):
    print("  pytorch_model.bin is not in this folder, so this section is skipped.")
    print("  It ships alongside model.safetensors in gpt2's repo if you want it.\n")
else:
    import re                                   # noqa: E402
    print(f"""  This folder has the SAME WEIGHTS TWICE, in two formats:

    model.safetensors   {os.path.getsize(WEIGHTS):>12,} bytes
    pytorch_model.bin   {os.path.getsize(LEGACY):>12,} bytes

  pytorch_model.bin is the old format: a Python PICKLE. And a pickle is not
  data -- it is a little program. Its opcodes include "import this module" and
  "call this function". Loading one runs those instructions.

  You can see the instructions sitting in the bytes. Searching the first 400 KB
  of each file for module and function names:
""")
    pat = rb"torch[._a-zA-Z0-9]*|OrderedDict|collections|_rebuild_tensor[_a-zA-Z0-9]*"
    for label, path in (("pytorch_model.bin", LEGACY),
                        ("model.safetensors", WEIGHTS)):
        found = sorted({h.decode("latin1")
                        for h in re.findall(pat, open(path, "rb").read(400_000))})
        print(f"    {label}")
        if found:
            for h in found:
                print(f"      {h}")
        else:
            print(f"      nothing. no module or function names anywhere.")
        print()

    print("""  That is the whole argument for safetensors. The .bin file names
  `torch._utils._rebuild_tensor_v2` and `collections.OrderedDict` because the
  loader must import and call them to rebuild the tensors. A file that can name
  one function can name any function, which is why `torch.load` on an untrusted
  .bin file is a genuine remote-code-execution risk, and why torch now defaults
  to `weights_only=True`.

  safetensors cannot do that. Its header is JSON -- offsets, shapes and dtypes.
  JSON has no way to express "call this". Reading it can only ever produce
  numbers.

  So the honest version of "a model is just a data file":

    model.safetensors   really is inert data
    pytorch_model.bin   is a program that mostly just rebuilds tensors

  Both hold identical weights. One of them you can safely open from a stranger.
""")


# ===========================================================================
print("=" * 74)
print("5. WHAT THE 160 TENSORS ARE")
print("=" * 74)

groups: dict[str, list[str]] = {}
for name in header:
    if name.startswith("h."):
        key = "h.N." + name.split(".", 2)[2]
    else:
        key = name
    groups.setdefault(key, []).append(name)

print(f"  {'tensor (N = block index)':<30} {'count':>5} {'shape':<14} {'numbers':>14}")
for key in sorted(groups, key=lambda k: -sum(
        math.prod(header[n]["shape"]) for n in groups[k])):
    names = groups[key]
    shape = tuple(header[names[0]]["shape"])
    tot = sum(math.prod(header[n]["shape"]) for n in names)
    print(f"  {key:<30} {len(names):>5} {str(shape):<14} {tot:>14,}")

all_numbers = sum(math.prod(v["shape"]) for v in header.values())
mask = sum(math.prod(header[n]["shape"]) for n in header if n.endswith(".attn.bias"))
print(f"  {'-' * 30} {'-' * 5} {'-' * 14} {'-' * 14}")
print(f"  {'everything in the file':<30} {len(header):>5} {'':<14} {all_numbers:>14,}")

print(f"""
  Two things in that table are worth stopping on.

  THE FILE HOLDS MORE THAN PARAMETERS. {all_numbers:,} numbers are stored, but
  GPT-2 has 124,439,808 parameters. The difference is exactly {all_numbers - 124439808:,},
  which is the `attn.bias` rows: 12 blocks x 1024 x 1024 = {mask:,}. Those are
  the causal mask -- a fixed triangle of 1s and 0s that stops a token attending
  to the future. It is data the model needs and never learns, so it is a
  BUFFER, not a parameter. Saved in the file either way.

  THERE IS NO lm_head.weight. The layer that turns 768 numbers into 50,257
  logits is missing from the file, because GPT-2 reuses `wte.weight` for it --
  the same matrix, transposed. That is called weight tying, it is declared
  nowhere in config.json, and the model CLASS knows to do it. Another thing
  the data cannot tell you and the engine has to.
""")


# ===========================================================================
print("=" * 74)
print("6. THE TOKENIZER IS TWO LOOKUP TABLES")
print("=" * 74)

vocab = json.load(open(os.path.join(SNAP, "vocab.json")))
merges = [l for l in open(os.path.join(SNAP, "merges.txt")).read().split("\n")
          if l.strip()]
inv = {v: k for k, v in vocab.items()}

print(f"  vocab.json  : {len(vocab):,} entries, token text -> id")
print(f"  merges.txt  : {len(merges) - 1:,} merge rules, after the "
      f"{merges[0]!r} header line\n")
print("  Five entries from vocab.json:\n")
for t in ["the", "Ġthe", "Ġbottle", "Ġcracked", "<|endoftext|>"]:
    print(f"    {t!r:<18} -> {vocab[t]}")
print("\n  The first five merge rules, in priority order:\n")
for line in merges[1:6]:
    print(f"    {line!r}")

print(f"""
  'Ġ' is how GPT-2 writes a leading space. Nothing clever -- a character
  chosen because it does not appear in normal text.

  Decoding needs no library. Invert the dict and index it:

    id 50256 -> {inv[50256]!r}
    id 11043 -> {inv[11043]!r}
    id  1886 -> {inv[1886]!r}
    id    62 -> {inv[62]!r}
    id  9186 -> {inv[9186]!r}

  Those five are the answer tokens from Block A. Put them together and you get
  {''.join(inv[i] for i in [11043, 1886, 62, 9186])!r}.

  Encoding is the other direction: split the text into characters, then apply
  the merge rules in priority order until nothing more merges. It is a sorting
  problem, not a learning problem. No weights are involved at any point.
""")


# ===========================================================================
print("=" * 74)
print("7. A LoRA ADAPTER IS THE SAME KIND OF FILE, 400x SMALLER")
print("=" * 74)

import torch                                   # noqa: E402
from safetensors.torch import save_file         # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_adapter")
os.makedirs(OUT, exist_ok=True)

R, ALPHA = 8, 16
torch.manual_seed(0)
tensors = {}
for layer in range(cfg["n_layer"]):
    p = f"base_model.model.transformer.h.{layer}.attn.c_attn"
    # peft's stored layout: lora_A is (r, in), lora_B is (out, r)
    tensors[f"{p}.lora_A.weight"] = torch.randn(R, cfg["n_embd"]) * 0.02
    tensors[f"{p}.lora_B.weight"] = torch.zeros(3 * cfg["n_embd"], R)

adapter_cfg = {
    "peft_type": "LORA", "task_type": "CAUSAL_LM",
    "base_model_name_or_path": "gpt2",
    "r": R, "lora_alpha": ALPHA, "lora_dropout": 0.0,
    "target_modules": ["c_attn"], "bias": "none", "fan_in_fan_out": True,
}
save_file(tensors, os.path.join(OUT, "adapter_model.safetensors"))
json.dump(adapter_cfg, open(os.path.join(OUT, "adapter_config.json"), "w"), indent=2)

a_size = os.path.getsize(os.path.join(OUT, "adapter_model.safetensors"))
c_size = os.path.getsize(os.path.join(OUT, "adapter_config.json"))
n_adapter = sum(t.numel() for t in tensors.values())

print(f"  Wrote one, in peft's exact format:\n")
print(f"    {OUT}/")
print(f"      adapter_model.safetensors   {a_size:>12,}  ({mb(a_size)})")
print(f"      adapter_config.json         {c_size:>12,}\n")
print(f"  {'tensors':<28} {len(tensors)}   ({cfg['n_layer']} layers x A and B)")
print(f"  {'numbers':<28} {n_adapter:,}")
print(f"  {'base model, for comparison':<28} {124439808:,}")
print(f"  {'adapter as % of base':<28} {n_adapter / 124439808 * 100:.3f}%")
print(f"  {'file size ratio':<28} {files[0][1] / a_size:.0f}x smaller\n")

with open(os.path.join(OUT, "adapter_model.safetensors"), "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    ah = json.loads(f.read(n))
ah.pop("__metadata__", None)
print("  Same header trick, on the adapter:\n")
for k in sorted(ah)[:4]:
    print(f"    {k:<58} {tuple(ah[k]['shape'])}")
print(f"    ... {len(ah) - 4} more\n")
print("  And its config:\n")
for line in json.dumps(adapter_cfg, indent=2).split("\n"):
    print(f"    {line}")

print(f"""
  That is the entire output of a LoRA fine-tune. Two matrices per adapted
  layer, and a config naming which layers were adapted and what r and alpha
  were. The base model's {files[0][1]:,} bytes are untouched and are not in here.

  Note `fan_in_fan_out: true`. GPT-2's Conv1D stores its weight as
  (in, out) instead of (out, in), and that flag is peft recording the fact so
  it transposes correctly at load time. The shapes above are peft's layout:
  lora_A is (r, in) = {tuple(ah[sorted(ah)[0]]['shape'])}, which is the transpose of the (in, r) we
  used when building it by hand in A4.
""")


# ===========================================================================
print("=" * 74)
print("8. DATA vs ENGINE, DEMONSTRATED")
print("=" * 74)
print("""  Everything above read metadata. Now read actual numbers out of the file
  and compute with them, using safetensors to read bytes and nothing else.
  No GPT2LMHeadModel, no config parsing, no model class.
""")

from safetensors import safe_open               # noqa: E402

with safe_open(WEIGHTS, framework="pt") as f:
    wte = f.get_tensor("wte.weight")            # token embedding table
    wpe = f.get_tensor("wpe.weight")            # position embedding table
    ln_w = f.get_tensor("h.0.ln_1.weight")
    ln_b = f.get_tensor("h.0.ln_1.bias")
    c_attn_w = f.get_tensor("h.0.attn.c_attn.weight")
    c_attn_b = f.get_tensor("h.0.attn.c_attn.bias")

print(f"  wte.weight  {tuple(wte.shape)}  the token embedding table")
print(f"  wpe.weight  {tuple(wpe.shape)}  the position embedding table\n")

tok_id = 11043                                  # 'dam', from Block A
print(f"  Step 1, embed token {tok_id} ({inv[tok_id]!r}) at position 0. This is a")
print(f"  ROW LOOKUP plus an add -- no matrix multiply yet:\n")
x = wte[tok_id] + wpe[0]
print(f"    wte[{tok_id}][:6]      = {[round(v, 4) for v in wte[tok_id][:6].tolist()]}")
print(f"    wpe[0][:6]          = {[round(v, 4) for v in wpe[0][:6].tolist()]}")
print(f"    sum[:6]             = {[round(v, 4) for v in x[:6].tolist()]}")
print(f"    shape               = {tuple(x.shape)}\n")

print("  Step 2, layer norm, then block 0's Q/K/V projection. NOW a matmul:\n")
mu, var = x.mean(), x.var(unbiased=False)
xn = (x - mu) / torch.sqrt(var + cfg["layer_norm_epsilon"]) * ln_w + ln_b
qkv = xn @ c_attn_w + c_attn_b
q, k, v = qkv.split(cfg["n_embd"])
print(f"    after layer norm    {tuple(xn.shape)}")
print(f"    @ c_attn.weight     {tuple(xn.shape)} @ {tuple(c_attn_w.shape)} -> {tuple(qkv.shape)}")
print(f"    split into Q, K, V  {tuple(q.shape)}  {tuple(k.shape)}  {tuple(v.shape)}")
print(f"    Q[:6]               = {[round(val, 4) for val in q[:6].tolist()]}\n")

print(f"  Is that right, though? Check it against the real engine.\n")

from transformers import AutoModelForCausalLM     # noqa: E402

engine = AutoModelForCausalLM.from_pretrained(
    "gpt2", attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0)
engine.eval()
grabbed = {}
engine.transformer.h[0].attn.c_attn.register_forward_hook(
    lambda mod, inp, out: grabbed.update(out=out))
with torch.no_grad():
    engine(input_ids=torch.tensor([[tok_id]]))
theirs = grabbed["out"][0, 0]

diff = (qkv - theirs).abs().max().item()
print(f"    our 4 lines of arithmetic : {[round(v, 4) for v in qkv[:6].tolist()]}")
print(f"    GPT2LMHeadModel           : {[round(v, 4) for v in theirs[:6].tolist()]}")
print(f"    max abs difference        : {diff:.3e}")
print(f"    allclose at 1e-5          : {torch.allclose(qkv, theirs, atol=1e-5)}")

print(f"""
  Identical to float rounding. We reimplemented a sliver of the engine and got
  the engine's answer, because the numbers came from the same file and we
  followed the same recipe.

  THAT is the division you were asking about.

    the file      gave us wte, wpe, ln_1.weight, c_attn.weight -- numbers
    we            supplied the lookup, the add, the normalisation, the matmul

  The file never said "now normalise, then multiply". We knew to do that
  because config.json said `GPT2LMHeadModel`, and that class's source code is
  the recipe. Swap in a different recipe and the same numbers produce garbage.

  A usable model = this data + an engine that knows the recipe. transformers,
  llama.cpp and vLLM are three different engines for the same kind of data.

  We did ONE token through ONE sixth of ONE block out of twelve. S3 writes the
  whole loop properly, and that is where the KV cache will come from.
""")

print("=" * 74)
print("NEXT: s2 -- run a LoRA fine-tune, then merge the adapter by hand and")
print("      watch W + (alpha/r) * B @ A turn two files into one.")
print("=" * 74)
