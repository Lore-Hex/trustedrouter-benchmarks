"""Splice glm-5.2 zai responses into the SQV panel, replacing broken default-route responses."""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

panel_path = HERE / "results" / "simpleqa_verified_panel.json"
zai_path = HERE / "results" / "sqv_glm52_zai.json"

panel = json.loads(panel_path.read_text(encoding="utf-8"))
zai = json.loads(zai_path.read_text(encoding="utf-8"))

# Build lookup from question id → zai response
zai_by_id = {r["id"]: r for r in zai["responses"]}

replaced = 0
kept = 0
new_responses = []
for r in panel["responses"]:
    if r["model"] == "z-ai/glm-5.2":
        if r["id"] in zai_by_id:
            new_responses.append(zai_by_id[r["id"]])
            replaced += 1
        else:
            new_responses.append(r)
            kept += 1
    else:
        new_responses.append(r)

print(f"glm-5.2: replaced {replaced} responses, kept {kept} (no zai match)")
print(f"total responses: {len(new_responses)}")

panel["responses"] = new_responses
panel_path.write_text(json.dumps(panel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"wrote {panel_path}")
