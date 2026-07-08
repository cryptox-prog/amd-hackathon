import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "accounts/fireworks/models/minimax-m3"
SYSTEM_PROMPT = (
    "You are a precise assistant. "
    "Answer accurately and concisely. "
    "Return only the requested answer. "
    "Use as few tokens as possible."
)


def get_model() -> str:
    allowed_models = os.getenv("ALLOWED_MODELS", "")
    if allowed_models.strip():
        return allowed_models.split(",")[0].strip()
    return DEFAULT_MODEL


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ["FIREWORKS_BASE_URL"],
        timeout=30,
    )


def get_response(message: str, model: str | None = None) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=model or get_model(),
        max_tokens=10000,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": message,
            },
        ],
    )

    return (response.choices[0].message.content or "").strip()


def get_answers(tasks: list[dict[str, Any]], model: str | None = None) -> list[dict[str, str]]:
    results = []

    for task in tasks:
        results.append(
            {
                "task_id": task["task_id"],
                "answer": get_response(task["prompt"], model=model),
            }
        )

    return results
