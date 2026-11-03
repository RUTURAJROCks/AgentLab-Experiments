# agentlab
Eight demo agents built from aiengineeringfromscratch.com concepts, one OpenRouter client so every US and Chinese frontier model is a string swap.
1. export OPENROUTER_API_KEY=sk-or-...
2. pip install -r requirements.txt
3. python -c "from agentlab import react; print(react.run('What is 17*23 + 4?')[0])"
Claude Code: claude mcp add agentlab -e OPENROUTER_API_KEY=$OPENROUTER_API_KEY -- python $(pwd)/server.py
Codex: add [mcp_servers.agentlab] to ~/.codex/config.toml (see site).
Colab: open agentlab_colab.ipynb.

## Eval projects
- evals/decision-models: Sarvam Prism vs Jev vs other decision models (see its README and SOURCES.md)

## Site
- site/index.html is the agentlab page (deployed on Vercel).
