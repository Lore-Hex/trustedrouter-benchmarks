"""Re-generate the verbose reasoning models that truncated at max_tokens=8192 on
the SimpleQA Verified panel, at a generous 32768 budget, and splice the fresh
answers back into results/simpleqa_verified_panel.json (replacing their rows).

One-off maintenance script — see the run.py docstring/comment for why the budget
matters. Run from the repo root with TRUSTEDROUTER_API_KEY set.
"""
from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from trbench import client
from trbench.evals.simpleqa_verified.loader import load

MODELS = ["z-ai/glm-5.1", "moonshotai/kimi-k2.6", "z-ai/glm-5", "z-ai/glm-5.2"]
MAX_TOKENS = 32768
PANEL = "results/simpleqa_verified_panel.json"


def main() -> int:
    key = client.api_key_from_env(os.environ.get("TRUSTEDROUTER_API_KEY"))
    bu = client.DEFAULT_BASE_URL
    items = load(limit=250)

    fresh: list[dict] = []
    for model in MODELS:
        def one(it: dict, model: str = model) -> dict:
            r = client.chat(
                base_url=bu, api_key=key, model=model,
                messages=[{"role": "user", "content": it["question"]}],
                max_tokens=MAX_TOKENS, temperature=0.0, timeout=240,
            )
            row = {"model": model, "id": it["id"], "question": it["question"], "target": it["target"]}
            if r.get("error"):
                row["error"] = r["error"]
            else:
                row["text"] = r.get("text", "")
            return row

        with ThreadPoolExecutor(max_workers=10) as pool:
            rows = list(pool.map(one, items))
        empty = sum(1 for r in rows if not (r.get("text") or "").strip() and not r.get("error"))
        errs = sum(1 for r in rows if r.get("error"))
        print(f"  {model}: {len(rows)} items, {empty} empty, {errs} errors", flush=True)
        fresh.extend(rows)

    d = json.loads(open(PANEL).read())
    keep = {m for m in MODELS}
    others = [r for r in d["responses"] if r.get("model") not in keep]
    d["responses"] = sorted(others + fresh, key=lambda r: (r["model"], str(r["id"])))
    with open(PANEL, "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
        f.write("\n")
    n_models = len({r["model"] for r in d["responses"]})
    print(f"spliced; panel now {len(d['responses'])} responses across {n_models} models", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
