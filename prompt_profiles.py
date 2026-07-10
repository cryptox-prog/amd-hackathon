import os
from dataclasses import dataclass

Category = str
Difficulty = str

MINIMAX = "accounts/fireworks/models/minimax-m3"
KIMI = "accounts/fireworks/models/kimi-k2p7-code"
GEMMA_26 = "accounts/fireworks/models/gemma-4-26b-a4b-it"
GEMMA_31 = "accounts/fireworks/models/gemma-4-31b-it"
GEMMA_31_NVFP4 = "accounts/fireworks/models/gemma-4-31b-it-nvfp4"


@dataclass(frozen=True)
class PromptProfile:
    category: Category
    difficulty: Difficulty
    model: str
    system_prompt: str
    output_style: str

    def messages_for(self, user_prompt: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": f"{self.system_prompt}\n\nOutput style: {self.output_style}",
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]


PROMPT_PROFILES: dict[Category, dict[Difficulty, PromptProfile]] = {
    "factual_qa": {
        "easy": PromptProfile(
            category="factual_qa",
            difficulty="easy",
            model=MINIMAX,
            system_prompt=(
                "Answer factual questions directly. Use only well-established facts. "
                "If the question has multiple parts, answer each part in order."
            ),
            output_style="Return one concise sentence. Do not add extra context.",
        ),
        "hard": PromptProfile(
            category="factual_qa",
            difficulty="hard",
            model=MINIMAX,
            system_prompt=(
                "Answer factual questions carefully and verify relationships between "
                "entities before responding. Avoid speculation."
            ),
            output_style="Return a compact answer with only the facts needed.",
        ),
    },
    "math_reasoning": {
        "easy": PromptProfile(
            category="math_reasoning",
            difficulty="easy",
            model=GEMMA_31_NVFP4,
            system_prompt=(
                "Solve arithmetic and word problems exactly. Track quantities and "
                "operations carefully before giving the final result."
            ),
            output_style="Return the final number with a brief calculation.",
        ),
        "hard": PromptProfile(
            category="math_reasoning",
            difficulty="hard",
            model=GEMMA_31_NVFP4,
            system_prompt=(
                "Solve multi-step reasoning problems exactly. Identify assumptions, "
                "perform calculations step by step internally, and check the final value."
            ),
            output_style="Return the final answer plus the shortest useful derivation.",
        ),
    },
    "sentiment": {
        "easy": PromptProfile(
            category="sentiment",
            difficulty="easy",
            model=MINIMAX,
            system_prompt=(
                "Classify the sentiment of the provided text. Consider mixed evidence "
                "when positive and negative statements both appear."
            ),
            output_style="Return one label: positive, negative, neutral, or mixed.",
        ),
        "hard": PromptProfile(
            category="sentiment",
            difficulty="hard",
            model=MINIMAX,
            system_prompt=(
                "Classify nuanced sentiment. Separate factual statements from opinions "
                "and account for contrastive wording."
            ),
            output_style="Return the label and a short reason.",
        ),
    },
    "summarization": {
        "easy": PromptProfile(
            category="summarization",
            difficulty="easy",
            model=MINIMAX,
            system_prompt=(
                "Summarize the source text faithfully. Preserve the central cause, "
                "constraint, or conclusion without adding outside information."
            ),
            output_style="Return exactly one sentence.",
        ),
        "hard": PromptProfile(
            category="summarization",
            difficulty="hard",
            model=MINIMAX,
            system_prompt=(
                "Compress dense technical text while preserving the main bottleneck, "
                "response, and constraints. Do not introduce unsupported claims."
            ),
            output_style="Return exactly one concise sentence.",
        ),
    },
    "ner": {
        "easy": PromptProfile(
            category="ner",
            difficulty="easy",
            model=GEMMA_31_NVFP4,
            system_prompt=(
                "Extract named entities from the text and assign clear entity types. "
                "Do not include generic nouns or unnamed references."
            ),
            output_style='Return JSON only, as [{"text": "...", "type": "..."}].',
        ),
        "hard": PromptProfile(
            category="ner",
            difficulty="hard",
            model=GEMMA_31_NVFP4,
            system_prompt=(
                "Extract named entities with precise types. Include people, "
                "organizations, locations, dates, products, and events when present."
            ),
            output_style='Return JSON only, as [{"text": "...", "type": "..."}].',
        ),
    },
    "debugging": {
        "easy": PromptProfile(
            category="debugging",
            difficulty="easy",
            model=KIMI,
            system_prompt=(
                "Find the bug in the provided code and provide the minimal corrected "
                "version. Prefer simple, idiomatic Python."
            ),
            output_style="Return the fixed code first, then one short explanation.",
        ),
        "hard": PromptProfile(
            category="debugging",
            difficulty="hard",
            model=KIMI,
            system_prompt=(
                "Diagnose code defects precisely. Consider edge cases, empty inputs, "
                "type assumptions, and expected behavior before proposing a fix."
            ),
            output_style="Return corrected code and a concise explanation of the bug.",
        ),
    },
    "logical_reasoning": {
        "easy": PromptProfile(
            category="logical_reasoning",
            difficulty="easy",
            model=MINIMAX,
            system_prompt=(
                "Solve logic puzzles by applying each constraint exactly. Eliminate "
                "impossible assignments before answering."
            ),
            output_style="Return the answer and one sentence explaining why.",
        ),
        "hard": PromptProfile(
            category="logical_reasoning",
            difficulty="hard",
            model=MINIMAX,
            system_prompt=(
                "Solve constraint reasoning tasks carefully. Track all entities, "
                "attributes, and exclusions before producing the final answer."
            ),
            output_style="Return the final answer with a compact reasoning trace.",
        ),
    },
    "code_generation": {
        "easy": PromptProfile(
            category="code_generation",
            difficulty="easy",
            model=KIMI,
            system_prompt=(
                "Write correct, simple Python code for the requested behavior. Include "
                "edge-case handling when the prompt asks for it."
            ),
            output_style="Return code only unless the prompt explicitly asks for explanation.",
        ),
        "hard": PromptProfile(
            category="code_generation",
            difficulty="hard",
            model=KIMI,
            system_prompt=(
                "Write robust Python code with clear handling of edge cases, duplicates, "
                "invalid inputs, and expected return values."
            ),
            output_style="Return code first, followed by minimal notes only if useful.",
        ),
    },
}

CATEGORIES = list(PROMPT_PROFILES)
DIFFICULTY = ["easy", "hard"]


def get_allowed_models() -> list[str]:
    models = os.getenv("ALLOWED_MODELS", "")
    return [model.strip() for model in models.split(",") if model.strip()]


def recommend_model(category: Category, difficulty: Difficulty = "easy") -> str:
    if category not in PROMPT_PROFILES:
        raise ValueError(f"Unknown category: {category}")
    if difficulty not in PROMPT_PROFILES[category]:
        raise ValueError(f"Unknown difficulty for {category}: {difficulty}")

    preferred_model = PROMPT_PROFILES[category][difficulty].model
    allowed_models = get_allowed_models()
    if not allowed_models or preferred_model in allowed_models:
        return preferred_model
    return allowed_models[0]


def get_prompt_profile(
    category: Category,
    difficulty: Difficulty = "easy",
) -> PromptProfile:
    profile = PROMPT_PROFILES[category][difficulty]
    return PromptProfile(
        category=profile.category,
        difficulty=profile.difficulty,
        model=recommend_model(category, difficulty),
        system_prompt=profile.system_prompt,
        output_style=profile.output_style,
    )


def build_messages(
    user_prompt: str,
    category: Category,
    difficulty: Difficulty = "easy",
) -> list[dict[str, str]]:
    return get_prompt_profile(category, difficulty).messages_for(user_prompt)
