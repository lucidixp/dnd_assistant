import os
import requests
from fastapi import FastAPI
from pathlib import Path

app = FastAPI()

VAULT_PATH = Path(os.getenv("VAULT_PATH", "/vault"))
OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL = os.getenv("MODEL", "qwen3:8b")

MAX_RESULTS = int(os.getenv("MAX_RESULTS", 5))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 6000))


# ----------------------------
# SIMPLE FILE SEARCH
# ----------------------------

def search_files(query: str):
    results = []

    query_lower = query.lower()

    for file in VAULT_PATH.rglob("*.md"):
        try:
            text = file.read_text(errors="ignore")

            if query_lower in text.lower():
                results.append({
                    "file": str(file),
                    "text": text[:2000]
                })

            if len(results) >= MAX_RESULTS:
                break

        except Exception:
            continue

    return results


# ----------------------------
# BUILD CONTEXT
# ----------------------------

def build_context(results):
    context = ""
    used = 0

    for r in results:
        chunk = f"\n\nSOURCE: {r['file']}\n{r['text']}\n"

        if used + len(chunk) > MAX_CONTEXT_CHARS:
            break

        context += chunk
        used += len(chunk)

    return context


# ----------------------------
# OLLAMA CALL
# ----------------------------

def ask_llm(question, context):
    prompt = f"""
You are a Dungeons & Dragons 5e rules assistant.

Only use the provided SRD context.
If the answer is not in the context, say so.

---

CONTEXT:
{context}

---

QUESTION:
{question}

---

ANSWER:
"""

    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    return r.json()["response"]


# ----------------------------
# API ENDPOINT
# ----------------------------

@app.get("/ask")
def ask(q: str):
    results = search_files(q)
    context = build_context(results)

    answer = ask_llm(q, context)

    return {
        "question": q,
        "answer": answer,
        "sources": [r["file"] for r in results]
    }


@app.get("/health")
def health():
    return {"status": "ok"}
