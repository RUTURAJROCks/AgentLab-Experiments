# AGENTS.md (read by Codex, and by Claude Code via CLAUDE.md)
- Python 3.10+. Install: pip install -r requirements.txt
- Secrets: OPENROUTER_API_KEY in env only. Never commit keys.
- Pick models with MODEL env or the model= argument (slugs from openrouter.ai/models).
- Before finishing any change: python -m py_compile agentlab/*.py and run the relevant agent once.
- Log every experiment to results.csv (date, agent, model, task, passed, tokens, seconds, usd).
