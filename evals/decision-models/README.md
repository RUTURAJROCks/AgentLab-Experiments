# eval project: Sarvam Prism vs Jev (decision models)
Steps
1. Keys: SARVAM_API_KEY (indus.sarvam.ai), OPENROUTER_API_KEY (openrouter.ai/settings/keys).
2. pip install -U "sarvamai>=0.1.43" requests
3. python run_eval.py  -> results.json (accuracy, error at 0.9 confidence, flips when shuffled, p50/p95 ms)
4. Grow tickets.jsonl to 200+ labelled rows (add Hindi, Marathi, Hinglish) before trusting any number.
5. Then run a public set (Banking77) and an agent-trace set; write up with the claims in SOURCES.md.
Prism weights need ~120 GB FP8 GPU memory, so free Colab uses the API, not local weights.
