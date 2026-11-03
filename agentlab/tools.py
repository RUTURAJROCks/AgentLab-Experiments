"""Tool registry: name -> (json schema, callable). Errors become observations,
never exceptions (Phase 14 lesson 01 / 06)."""
import json, inspect

REGISTRY = {}


def tool(fn):
    params = {n: {"type": "string"} for n in inspect.signature(fn).parameters}
    REGISTRY[fn.__name__] = (
        {"type": "function", "function": {
            "name": fn.__name__, "description": (fn.__doc__ or "").strip(),
            "parameters": {"type": "object", "properties": params,
                           "required": list(params)}}},
        fn)
    return fn


def schemas(names=None):
    return [s for n, (s, _) in REGISTRY.items() if names is None or n in names]


def dispatch(name, raw_args):
    if name not in REGISTRY:
        return f"error: unknown tool {name!r}"
    try:
        return str(REGISTRY[name][1](**json.loads(raw_args or "{}")))[:4000]
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


@tool
def calculator(expr):
    """Evaluate arithmetic like '(2+3)*4'."""
    if not set(expr) <= set("0123456789+-*/(). "):
        return "error: illegal character"
    return eval(expr, {"__builtins__": {}}, {})


@tool
def fetch(url):
    """Fetch a web page and return its text (first 4000 chars)."""
    import re, urllib.request
    html = urllib.request.urlopen(url, timeout=15).read().decode("utf8", "ignore")
    return re.sub(r"<[^>]+>", " ", html)
