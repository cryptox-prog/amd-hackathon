import os
import json
import sys
from openai import OpenAI

INPUT_PATH = "./input/tasks.json" # remove the . for docker test run???
OUTPUT_DIR = "./output"
OUTPUT_PATH = OUTPUT_DIR + "/results.json"

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ["FIREWORKS_API_KEY"]
BASE_URL = os.environ["FIREWORKS_BASE_URL"]
MODELS = os.environ["ALLOWED_MODELS"].split(",")

def get_tasks():
    tasks = None
    with open(INPUT_PATH, "r") as input_file:
        tasks = json.load(input_file)
    return tasks

def get_answers(client, model, tasks):
    results = []

    for task in tasks:
        response = client.chat.completions.create(
            model=model,
            max_tokens=1000000,
            temperature=1.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise assistant. "
                        "Answer accurately and concisely. "
                        "Return only the requested answer."
                        "Use as few tokens as possible."
                    ),
                },
                {
                    "role": "user",
                    "content": task["prompt"],
                },
            ],
        )

        answer = response.choices[0].message.content.strip()
        results.append(
            {
                "task_id": task["task_id"],
                "answer": answer,
            }
        )

    return results

def write_output(answers):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        json.load(f)

def main():
    model = "accounts/fireworks/models/minimax-m3" # MODELS[0].strip()
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=30
    )

    tasks = get_tasks()
    answers = get_answers(client, model, tasks)
    write_output(answers)

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)