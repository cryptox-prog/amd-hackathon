import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from router.infer_router import classify
from prompt_profiles import get_prompt_profile

load_dotenv()

def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ["FIREWORKS_BASE_URL"],
        timeout=30,
    )


def get_response(message: str) -> str:
    client = get_client()

    # -------------------------
    # Route the prompt
    # -------------------------
    route = classify(message)

    category = route["category"]
    difficulty = route["difficulty"]

    # -------------------------
    # Select model + prompts
    # -------------------------
    profile = get_prompt_profile(category, difficulty)

    response = client.chat.completions.create(
        model=profile.model,
        temperature=0.0,
        max_tokens=10000,
        messages=profile.messages_for(message),
    )

    return (response.choices[0].message.content or "").strip()


def get_answers(tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
    results = []

    for task in tasks:
        answer = get_response(task["prompt"])

        results.append(
            {
                "task_id": task["task_id"],
                "answer": answer,
            }
        )

    return results