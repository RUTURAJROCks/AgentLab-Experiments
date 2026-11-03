"""Agent 06: supervisor + workers + reviewer. The supervisor splits the job,
role workers draft, a reviewer approves or sends back. Lessons 16.05, 14.39,
16.23 (watch for groupthink: give the reviewer a different model)."""
import json, re
from .llm import ask

ROLES = {"researcher": "You gather facts and cite sources.",
         "writer": "You write clear, concrete prose.",
         "critic": "You find the weakest claim and fix it."}


def run(goal, reviewer_model=None, rounds=2):
    plan = ask(f"Goal: {goal}\nRoles: {list(ROLES)}\nReturn JSON "
               '[{"role":..., "task":...}] covering the goal.',
               system="You are a supervisor. JSON only.")
    jobs = json.loads(re.search(r"\[.*\]", plan, re.S).group(0))
    draft = ""
    for j in jobs:
        draft += f"\n\n## {j['role']}\n" + ask(
            f"{j['task']}\n\nWork so far:{draft[-4000:]}", system=ROLES[j["role"]])
    for _ in range(rounds):
        verdict = ask(f"Goal: {goal}\nDraft:\n{draft[-6000:]}\nReply APPROVE or "
                      "list concrete fixes.", model=reviewer_model)
        if verdict.strip().upper().startswith("APPROVE"):
            break
        draft = ask(f"Apply these fixes:\n{verdict}\n\nDraft:\n{draft[-6000:]}")
    return draft
