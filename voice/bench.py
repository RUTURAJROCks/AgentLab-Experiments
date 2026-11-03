"""Score the voice agent on VoiceBench (TACL 2026, github.com/MatthewCYM/VoiceBench)
with the benchmark's OWN scorers, so numbers line up with its leaderboard.

  python -m voice.bench --mode hybrid --subsets openbookqa ifeval advbench bbh --limit 100

Writes <mode>-<subset>-audio.jsonl in VoiceBench's format (each row + 'response')
and runs VoiceBench/evaluate.py on it. Also records time to first token and full
reply time. Subsets that need a GPT judge (alpacaeval_full, commoneval, wildvoice,
sd-qa) are generated here; score them with VoiceBench's api_judge.py and an
OpenAI key if you want the full overall score.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

EVALUATOR = {"openbookqa": "mcq", "mmsu": "mcq", "ifeval": "ifeval",
             "advbench": "harm", "bbh": "bbh", "sd-qa": "qa",
             "alpacaeval_full": "open", "commoneval": "open", "wildvoice": "open"}
THINK = {"openbookqa": True, "mmsu": True, "bbh": True}   # reasoning sets
VB = os.getenv("VOICEBENCH_DIR", "VoiceBench")


def run(mode, subsets, limit, think_override=None):
    from datasets import load_dataset, Audio
    from .agent import VoiceAgent
    agent = VoiceAgent(mode=mode)
    report = {}
    for sub in subsets:
        split = "usa" if sub == "sd-qa" else "test"
        ds = load_dataset("hlt-lab/voicebench", sub, split=split).cast_column(
            "audio", Audio(sampling_rate=16000))
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
        think = THINK.get(sub, False) if think_override is None else think_override
        rows, secs = [], []
        for item in ds:
            t = time.perf_counter()
            reply, transcript = agent.reply(audio=np.asarray(item["audio"]["array"], np.float32),
                                            sr=16000, think=think, keep_history=False)
            secs.append(time.perf_counter() - t)
            row = {k: v for k, v in item.items() if k != "audio"}
            row.update(response=reply, transcript=transcript)
            rows.append(row)
        path = f"{mode}-{sub}-{split}-audio.jsonl"
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        score = None
        if EVALUATOR[sub] != "open" and sub != "sd-qa":
            res = subprocess.run([sys.executable, "evaluate.py", "--src_file",
                                  os.path.abspath(path), "--evaluator", EVALUATOR[sub]],
                                 cwd=VB, capture_output=True, text=True)
            score = (res.stdout + res.stderr).strip().splitlines()[-1:]
        report[sub] = {"n": len(rows), "think": think, "score": score,
                       "p50_reply_s": round(float(np.median(secs)), 2),
                       "p95_reply_s": round(float(np.percentile(secs, 95)), 2), "file": path}
        print(sub, report[sub], flush=True)
    json.dump(report, open(f"report-{mode}.json", "w"), indent=2)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="hybrid", choices=["native", "cascade", "hybrid"])
    p.add_argument("--subsets", nargs="+", default=["openbookqa", "ifeval", "advbench", "bbh"])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--think", choices=["on", "off"], default=None)
    a = p.parse_args()
    run(a.mode, a.subsets, a.limit, None if a.think is None else a.think == "on")
