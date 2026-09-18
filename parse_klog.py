"""Print the tail of a Kaggle kernel log in readable form."""
import json
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "kaggle_out/pec2-world-models-run.log"
raw = open(path, encoding="utf-8", errors="replace").read()
print("log bytes:", len(raw))

pat = re.compile(r'\{"stream_name":"(\w+)","time":([0-9.]+),"data":"((?:[^"\\]|\\.)*)"\}')
out = []
for stream, t, data in pat.findall(raw):
    try:
        text = json.loads('"' + data + '"')
    except json.JSONDecodeError:
        text = data
    out.append((stream, float(t), text))

print("records:", len(out))
for stream, t, text in out[-45:]:
    print(f"[{stream} {t:8.2f}] {text}", end="")
