import os
import json
import sys

from main import get_answers

INPUT_PATH = "./input/tasks.json" # remove the . for docker test run???
OUTPUT_DIR = "./output"
OUTPUT_PATH = OUTPUT_DIR + "/results.json"

def get_tasks():
    tasks = None
    with open(INPUT_PATH, "r") as input_file:
        tasks = json.load(input_file)
    return tasks

def write_output(answers):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        json.load(f)

def main():
    tasks = get_tasks()
    answers = get_answers(tasks)
    write_output(answers)

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
