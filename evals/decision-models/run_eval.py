"""Sarvam Prism vs Jev vs other decision models vs an LLM baseline.
Measures: accuracy, calibration (expected vs actual error at >=0.9),
order invariance (shuffle options), p50/p95 latency.
Usage: python run_eval.py   (edit MODELS below; skip ones you have no key for)"""
import json, random, statistics as st
from decide import decide

MODELS = [  # (label, backend, model id)
    ("Sarvam Prism", "sarvam", "sarvam-prism"),
    ("Jev 1.14", "openrouter", "typesafe/jev-1.14"),
    ("Microsoft-Decision-1", "openrouter", "microsoft/microsoft-decision-1"),
    ("Clef", "openrouter", "cloudflare/clef"),
    ("LLM baseline", "llm", "openrouter/auto"),
]
TEAMS = {"billing": "payments, charges, invoices and refunds",
         "tech": "app, website, login or API problems",
         "delivery": "shipping, tracking and damaged parcels"}


def questions(order):
    return {"team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                     "criteria": {k: TEAMS[k] for k in order}},
            "urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?"}}


def evaluate(label, backend, model, rows):
    hits, conf_hits, conf_n, flips, lat = 0, 0, 0, 0, []
    for r in rows:
        a, ms = decide(backend, model, r["state"], questions(list(TEAMS)))
        lat.append(ms)
        ok = a["team"]["top"] == r["team"] and a["urgent"]["top"] == r["urgent"]
        hits += ok
        p = a["team"]["probs"].get(a["team"]["top"], 0)
        if p >= 0.9:
            conf_n += 1; conf_hits += a["team"]["top"] == r["team"]
        shuffled = list(TEAMS); random.shuffle(shuffled)
        b, _ = decide(backend, model, r["state"], questions(shuffled))
        flips += b["team"]["top"] != a["team"]["top"]
    n = len(rows); lat.sort()
    return {"model": label, "accuracy": round(hits / n, 3),
            "answers_at_0.9": f"{conf_n}/{n}",
            "error_at_0.9": round(1 - conf_hits / conf_n, 3) if conf_n else None,
            "flips_when_shuffled": f"{flips}/{n}",
            "p50_ms": round(st.median(lat)), "p95_ms": round(lat[int(0.95 * (n - 1))])}


if __name__ == "__main__":
    rows = [json.loads(l) for l in open("tickets.jsonl")]
    results = []
    for m in MODELS:
        try:
            results.append(evaluate(*m, rows)); print(results[-1])
        except Exception as e:
            print(m[0], "skipped:", e)
    json.dump(results, open("results.json", "w"), indent=2)
