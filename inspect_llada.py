import json, os, glob, re
d = os.path.expanduser("~/dflex_proj/llada2mini")
cfg = json.load(open(os.path.join(d, "config.json")))
print("ARCH", cfg.get("architectures"), "| type", cfg.get("model_type"),
      "| vocab", cfg.get("vocab_size"), "| layers", cfg.get("num_hidden_layers"))
print("token cfg:", {k: v for k, v in cfg.items() if "mask" in k.lower() or "token_id" in k.lower()})
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
print("mask_token", tok.mask_token, "| mask_id", tok.mask_token_id,
      "| pad_id", tok.pad_token_id, "| len", len(tok))
print("special", tok.special_tokens_map)
names = set()
for py in glob.glob(os.path.join(d, "*.py")):
    for m in re.finditer(r"self\.(\w+)\s*=\s*nn\.Linear", open(py).read()):
        names.add(m.group(1))
print("Linear names:", sorted(names))
