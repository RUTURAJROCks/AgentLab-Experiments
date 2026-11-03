"""Agent 04: hybrid memory (Mem0-style). Extract facts after each turn,
store with embeddings, recall the top-k before the next one. Lessons
14.07-14.09. Embeddings run locally, so it works free on Colab."""
import json, os
import numpy as np
from sentence_transformers import SentenceTransformer
from .llm import ask

_emb = SentenceTransformer("all-MiniLM-L6-v2")


class Memory:
    def __init__(self, path="memory.json"):
        self.path = path
        self.facts = json.load(open(path)) if os.path.exists(path) else []

    def add(self, fact):
        self.facts.append(fact)
        json.dump(self.facts, open(self.path, "w"))

    def recall(self, query, k=5):
        if not self.facts:
            return []
        v = _emb.encode([query] + self.facts, normalize_embeddings=True)
        order = np.argsort(-(v[1:] @ v[0]))[:k]
        return [self.facts[i] for i in order]


def chat_turn(mem, user_msg):
    context = "\n".join(mem.recall(user_msg)) or "nothing yet"
    reply = ask(f"What you remember about the user:\n{context}\n\nUser: {user_msg}",
                system="You are a helpful assistant with long-term memory.")
    facts = ask(f"User said: {user_msg}\nList new durable facts about the user, "
                "one per line, or NONE.")
    for f in facts.splitlines():
        if f.strip() and f.strip().upper() != "NONE":
            mem.add(f.strip("- ").strip())
    return reply
