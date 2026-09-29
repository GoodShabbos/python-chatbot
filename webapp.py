import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from flask import Flask, jsonify, request, send_file


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11111").rstrip("/")
if "://" not in OLLAMA_HOST:
    OLLAMA_HOST = "http://" + OLLAMA_HOST

MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud")
PROMPT_FILE = Path(__file__).with_name("system_prompt.txt")

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent


def _serve(filename):
    path = BASE_DIR / filename
    if filename == "index.html":
        html = path.read_text(encoding="utf-8")
        return html.replace("{{ model }}", MODEL)
    return send_file(path)


@app.get("/")
def home():
    return _serve("index.html")


@app.get("/chat")
def chat_page():
    return _serve("index.html")


@app.get("/style.css")
def style_css():
    return _serve("style.css")


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    override = data.get("system_prompt", "")

    if not isinstance(override, str):
        return jsonify(error="System prompt must be text."), 400
    if len(override) > 20000:
        return jsonify(error="System prompt is too long."), 400

    if override.strip():
        system_context = override.strip()
    else:
        try:
            system_context = PROMPT_FILE.read_text(encoding="utf-8").strip()
        except OSError as error:
            return jsonify(error=f"Could not read {PROMPT_FILE.name}: {error}"), 500

    if not system_context:
        return jsonify(error=f"{PROMPT_FILE.name} is empty."), 500

    chat_history = data.get("messages")
    if not isinstance(chat_history, list):
        return jsonify(error="Invalid chat history."), 400

    messages = [{"role": "system", "content": system_context}]

    for item in chat_history[-20:]:
        if not isinstance(item, dict):
            return jsonify(error="Invalid chat history."), 400

        role = item.get("role")
        content = item.get("content")

        if role not in {"user", "assistant"} or not isinstance(content, str):
            return jsonify(error="Invalid chat message."), 400

        messages.append({"role": role, "content": content[:12000]})

    if messages[-1]["role"] != "user":
        return jsonify(error="Send a message to continue."), 400

    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }).encode("utf-8")

    ollama_request = Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(ollama_request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
        return jsonify(reply=result["message"]["content"])
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        return jsonify(error=f"Ollama error ({error.code}): {details}"), 502
    except (URLError, TimeoutError) as error:
        return jsonify(error=f"Could not connect to Ollama: {error}"), 502
    except (KeyError, json.JSONDecodeError):
        return jsonify(error="Ollama returned an invalid response."), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)