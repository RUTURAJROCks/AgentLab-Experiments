"""One client for every lab. OpenRouter speaks the OpenAI API, so swapping
Claude, GPT, Gemini, DeepSeek, Qwen, Kimi, GLM or MiniMax is one string."""
import os, time
from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.environ["OPENROUTER_API_KEY"],
)
MODEL = os.getenv("MODEL", "openrouter/auto")
USAGE = {"calls": 0, "in": 0, "out": 0, "seconds": 0.0}


def chat(messages, tools=None, model=None, **kw):
    t = time.time()
    args = dict(model=model or MODEL, messages=messages, **kw)
    if tools:
        args["tools"] = tools
    r = client.chat.completions.create(**args)
    USAGE["calls"] += 1
    USAGE["seconds"] += time.time() - t
    if r.usage:
        USAGE["in"] += r.usage.prompt_tokens
        USAGE["out"] += r.usage.completion_tokens
    return r.choices[0].message


def ask(prompt, system="You are precise and brief.", model=None):
    return chat([{"role": "system", "content": system},
                 {"role": "user", "content": prompt}], model=model).content or ""
