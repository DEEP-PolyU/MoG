"""Remove the 'timestamp' field from every line in all .jsonl files under results/."""

import json
import glob
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

for path in sorted(glob.glob(os.path.join(script_dir, "**", "*.jsonl"), recursive=True)):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "summary":
                continue
            obj.pop("timestamp", None)
            lines.append(json.dumps(obj, ensure_ascii=False))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"✓ {os.path.relpath(path, script_dir)}  ({len(lines)} lines)")

print("Done.")
