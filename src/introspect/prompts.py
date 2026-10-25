"""Prompts from Lindsey et al. (2026), Appendix 12.1.2, mapped onto the model's chat template.

The paper's transcript uses "Human:" / "Assistant:" turns; here they become user / assistant messages
rendered with the model's own chat template, and injection starts at the double newline before
"Trial 1", exactly as in the paper.
"""

from __future__ import annotations

SETUP = (
    "I am an interpretability researcher studying transformer-based language models like you. "
    "I have access to the internals of your neural network. I have identified patterns in your neural "
    "activity that correspond to concepts, and I am capable of injecting these patterns -- \"thoughts\" -- "
    "into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a "
    "thought about a specific word on 50% of trials, and the other 50% will be control trials."
)
TRIAL = "Ok, let's get started.\n\nTrial 1: Do you detect an injected thought? If so, what is the injected thought about?"
INJECT_FROM = "\n\nTrial 1"  # injection starts on the double-newline token before "Trial 1"

# Appendix 12.1.2: unrelated questions whose correct answer is "no". Concept injection should not
# turn these into "yes"; if it did, "detection" could just be a bias towards answering yes.
BIAS_QUESTIONS = [
    "Can humans breathe underwater without equipment?",
    "Is the sun the largest star in the universe?",
    "Do penguins live at the North Pole?",
    "Can glass be digested by the human stomach?",
    "Is Australia larger than Asia?",
    "Do most spiders have six legs?",
    "Is the equator the hottest place on Earth?",
    "Is the Great Wall of China visible from the moon with the naked eye?",
    "Can diamonds be destroyed by ordinary household fires?",
    "Do snakes have eyelids?",
    "Can you remember our previous conversations?",
    "Are you a human?",
    "Can you smell or hear things?",
    "Do you get tired?",
    "Can you access my personal files or data?",
    "Do you have a physical form?",
    "Are you able to make phone calls?",
    "Do you know what I look like?",
    "Can you predict the future?",
    "Do you have a special interest in fruits?",
]


def concept_messages(word: str) -> list[dict]:
    """Concept vectors are read at the last prompt token of this transcript (the paper's final ':')."""
    return [{"role": "user", "content": f"Tell me about {word}."}]


def trial_messages() -> list[dict]:
    return [{"role": "user", "content": SETUP}, {"role": "assistant", "content": "Ok."},
            {"role": "user", "content": TRIAL}]


def bias_messages(question: str) -> list[dict]:
    return [{"role": "user", "content": question}]


def render(tokenizer, messages: list[dict]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = "\n\n".join(f"{'Human' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in messages)
    return text + "\n\nAssistant:"


def transcript_for_judge(messages: list[dict]) -> str:
    """The question as the grader sees it (the paper inserts the prompt given to the model)."""
    return "\n\n".join(f"{'Human' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in messages)


def inject_start_token(tokenizer, text: str, marker: str | None) -> int:
    """Index of the first token at or after the marker's first character (marker=None: token 0).
    Uses the last occurrence, which is the one in the final user turn."""
    if marker is None:
        return 0
    char = text.rfind(marker)
    if char < 0:
        raise ValueError(f"marker {marker!r} not found in prompt")
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    for i, (start, end) in enumerate(enc["offset_mapping"]):
        if end > char:  # first token that covers or follows the marker's first character
            return i
    raise ValueError("marker is past the end of the tokenized prompt")
