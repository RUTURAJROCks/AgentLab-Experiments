"""Agent 02: plan-and-execute (ReWOO). The planner writes every step up front
with #E placeholders, workers fill them, a solver writes the answer. Fewer
LLM calls than ReAct. Lesson 14.02."""
import json, re
from .llm import ask
from .tools import dispatch, REGISTRY


def run(task):
    plan = ask(
        f"Task: {task}\nTools: {list(REGISTRY)}\n"
        'Return JSON only: [{"id":"E1","tool":"name","args":{...}}, ...]. '
        "Args may reference earlier results as #E1.",
        system="You are a planner. Output valid JSON only.")
    steps = json.loads(re.search(r"\[.*\]", plan, re.S).group(0))
    evidence = {}
    for s in steps:
        args = json.dumps(s["args"])
        for k, v in evidence.items():
            args = args.replace(f"#{k}", json.dumps(v)[1:-1])
        evidence[s["id"]] = dispatch(s["tool"], args)
    return ask(f"Task: {task}\nEvidence: {json.dumps(evidence)[:6000]}\n"
               "Write the final answer."), steps, evidence
