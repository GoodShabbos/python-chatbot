import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11111").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:20b-cloud")
FILTER_FILE = Path(__file__).with_name("system_prompt.txt")


def main():
    try:
        system_context = FILTER_FILE.read_text(encoding="utf-8").strip()
    except OSError as error:
        print(f"Could not read {FILTER_FILE.name}: {error}")
        return

    if not system_context:
        print(f"{FILTER_FILE.name} is empty.")
        return

    messages = [{"role": "system", "content": system_context}]
    print(f"Welcome to Jeff's Chat!" f" Chatting with Jeff, who is powered by {MODEL}. Type /exit to quit.") #this starts the convo

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_text.lower() in {"/exit", "exit", "quit"}:
            break
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        payload = json.dumps({
            "model": MODEL,
            "messages": messages,
            "stream": False,
        }).encode("utf-8")

        request = Request(
            f"{OLLAMA_HOST}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
            answer = result["message"]["content"]
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            print(f"Ollama returned an error ({error.code}): {details}")
            messages.pop()
            continue
        except (URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
            print(f"Could not get a response from Ollama: {error}")
            messages.pop()
            continue

        print(f"Jeff: {answer}")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()