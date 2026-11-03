"""One interface for System One decision models.
backend "sarvam"     -> Sarvam Prism via the Sarvam API  (needs SARVAM_API_KEY)
backend "openrouter" -> Jev or any decision model on OpenRouter's Decisions API
                        (typesafe/jev-1.14, microsoft/microsoft-decision-1,
                        cloudflare/clef, perplexity/..., needs OPENROUTER_API_KEY)
backend "llm"        -> baseline: a normal chat LLM asked to pick a label
Every backend returns {question_id: {"type", "top", "probs"}} so they score
the same way."""
import json, os, re, time, requests

OR_URL = "https://openrouter.ai/api/alpha/decisions"


def _norm(answers):
    out = {}
    for qid, a in answers.items():
        a = a if isinstance(a, dict) else (a.model_dump() if hasattr(a, "model_dump") else vars(a))
        t = a.get("type")
        if t in ("noul", "yes_no"):
            p = a.get("noul", a.get("yes"))
            out[qid] = {"type": "noul", "top": p >= 0.5, "probs": {"yes": p, "no": 1 - p}}
        else:
            probs = a.get("probabilities") or {}
            top = a.get("choice") if t == "choice" else max(probs, key=probs.get)
            out[qid] = {"type": t, "top": top, "probs": probs}
    return out


def decide(backend, model, state, questions):
    t = time.perf_counter()
    if backend == "sarvam":
        from sarvamai import SarvamAI
        c = SarvamAI(api_subscription_key=os.environ["SARVAM_API_KEY"])
        r = c.decisions.create(model=model, state=state, questions=questions)
        ans = r.answers
    elif backend == "openrouter":
        r = requests.post(OR_URL, timeout=60, json={"model": model, "state": state, "questions": questions},
                          headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
        r.raise_for_status()
        ans = r.json()["answers"]
    elif backend == "llm":
        ans = _llm(model, state, questions)
    else:
        raise ValueError(backend)
    return _norm(ans), (time.perf_counter() - t) * 1000


def _llm(model, state, questions):
    """Baseline the decision models are meant to replace: prompt + parse."""
    spec = {q: {"instructions": v["instructions"],
                "options": list(v["criteria"]) if v["type"] == "choice" else ["yes", "no"]}
            for q, v in questions.items()}
    prompt = (f"State:\n{json.dumps(state)}\n\nAnswer each question with one option. "
              f"Questions: {json.dumps(spec)}\nReturn JSON {{question_id: option}} only.")
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=60,
                      headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                      json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0})
    r.raise_for_status()
    picks = json.loads(re.search(r"\{.*\}", r.json()["choices"][0]["message"]["content"], re.S).group(0))
    out = {}
    for q, v in questions.items():
        p = str(picks.get(q, "")).lower()
        if v["type"] == "noul":
            out[q] = {"type": "noul", "noul": 1.0 if p.startswith("y") else 0.0}
        else:
            out[q] = {"type": "choice", "choice": p, "probabilities": {k: float(k == p) for k in v["criteria"]}}
    return out
