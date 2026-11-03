"""Agent 05: harnessed coding agent. Scope contract + sandbox denylist +
verification gate: the agent is done only when pytest passes, never when it
says so. Lessons 14.31-14.42, capstones 19.20-19.27."""
import pathlib, subprocess
from .llm import ask

DENY = ("rm -rf", "curl ", "wget ", "sudo", "os.system", "subprocess", "shutil.rmtree")


def gate(workdir):
    r = subprocess.run(["python", "-m", "pytest", "-q", "-x"], cwd=workdir,
                       capture_output=True, text=True, timeout=120)
    return r.returncode == 0, (r.stdout + r.stderr)[-2500:]


def run(spec, workdir="work", target="solution.py", max_iters=5):
    wd = pathlib.Path(workdir)
    tests = (wd / "test_solution.py").read_text()
    code, feedback = "", "first attempt"
    for i in range(max_iters):
        code = ask(f"Spec:\n{spec}\n\nTests:\n{tests}\n\nPrevious code:\n{code}\n"
                   f"Feedback:\n{feedback}\n\nReturn the full {target} only, no fences.",
                   system="You only edit the one file you are given.")
        code = code.replace("```python", "").replace("```", "").strip()
        if any(d in code for d in DENY):          # sandbox rule
            feedback = "rejected: uses a denied operation"
            continue
        (wd / target).write_text(code)
        ok, feedback = gate(wd)                   # verification gate
        print(f"iter {i + 1}: {'PASS' if ok else 'FAIL'}")
        if ok:
            return True, i + 1
    return False, max_iters
