"""Agent 08: eval harness. Same tasks, many models: full completion rate,
tokens, seconds, and cost per COMPLETED task. Lessons 14.30, 19.27.
Use it for the US vs China rounds."""
import time
from . import llm

TASKS = [  # (prompt, checker)
    ("What is 17*23? Reply with the number only.", lambda a: a.strip() == "391"),
    ("Return the JSON {\"ok\": true} and nothing else.",
     lambda a: a.strip().strip("`").replace("json", "").strip() == '{"ok": true}'),
    ("Name the capital of Australia in one word.", lambda a: "canberra" in a.lower()),
]


def bench(models, tasks=TASKS, price_per_mtok=None):
    """price_per_mtok: {model: (input_usd, output_usd)} from openrouter.ai/models"""
    rows = []
    for m in models:
        before = dict(llm.USAGE)
        t, passed = time.time(), 0
        for prompt, check in tasks:
            try:
                passed += bool(check(llm.ask(prompt, model=m)))
            except Exception as e:
                print(m, "error", e)
        tin, tout = llm.USAGE["in"] - before["in"], llm.USAGE["out"] - before["out"]
        pin, pout = (price_per_mtok or {}).get(m, (0, 0))
        cost = (tin * pin + tout * pout) / 1e6
        rows.append({"model": m, "passed": f"{passed}/{len(tasks)}",
                     "tokens": tin + tout, "seconds": round(time.time() - t, 1),
                     "usd_per_completed": round(cost / passed, 6) if passed else None})
    return rows
