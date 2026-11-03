"""Entry point for Claude Code / Codex: python /abs/path/agentlab/server.py"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from agentlab.mcp_server import mcp

if __name__ == "__main__":
    mcp.run()
