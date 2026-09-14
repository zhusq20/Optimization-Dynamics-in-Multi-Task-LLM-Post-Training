import re
import string
from collections.abc import Iterable

DEFAULT_VALID_LETTERS = list(string.ascii_uppercase[:10])
GPQA_SCORER_VERSION = "final-answer-v2"

_FINAL_ANSWER = re.compile(r"\bfinal\s+(?:answer|option|choice)\b\s*(?:is\b\s*)?[:=]?\s*", re.IGNORECASE)
_NAMED_ANSWER = re.compile(
    r"\b(?:answer\b\s*(?:is\b\s*)?[:=]?\s*|(?:option|choice)\b\s*(?:is\b\s*[:=]?\s*|[:=]\s*))",
    re.IGNORECASE,
)


def _strip_chain_of_thought(text: str) -> str:
    if not text:
        return ""

    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    if "<think>" in text:
        return ""

    return text


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _answer_text(response: str) -> str:
    text = _strip_chain_of_thought(response)
    text = re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>)\s*\Z", "", text)
    return text.replace("**", "").replace("`", "").strip()


def _option_at_start(text: str) -> str | None:
    text = text.replace(r"\(", "").replace(r"\)", "").replace("$", "")
    text = re.sub(r"\\(?:boxed|text|mathrm)\s*\{", "", text).lstrip(" \t\r\n({")
    match = re.match(r"([A-Z])(?!\w)", text, re.IGNORECASE)
    if match is None:
        return None
    remainder = text[match.end():].lstrip(" \t)}")
    if re.match(r"(?:/|,|\bor\b|\band\b)\s*\(?[A-Z](?!\w)", remainder, re.IGNORECASE):
        return None
    return match.group(1).upper()


def _extract_letter_from_response(response: str, valid_letters: Iterable[str]) -> str | None:
    """Read an explicit answer, boxed option, or final standalone option.

    Explicit final answers take precedence over earlier reasoning. Arbitrary
    option mentions in the response body are not answer selections.
    """
    if not response:
        return None

    text = _answer_text(response)
    valid_letters = {letter.upper() for letter in valid_letters}
    final = list(_FINAL_ANSWER.finditer(text))
    if final:
        final_text = text[final[-1].end():]
        letter = _option_at_start(final_text)
        # A Markdown "Final Answer" section may contain prose followed by
        # another explicit "Answer: C" line. Never fall back outside it.
        if letter is None:
            nested = list(_NAMED_ANSWER.finditer(final_text))
            if nested:
                letter = _option_at_start(final_text[nested[-1].end():])
        return letter if letter in valid_letters else None

    candidates = []
    for match in _NAMED_ANSWER.finditer(text):
        letter = _option_at_start(text[match.end():])
        if letter is not None:
            candidates.append((match.start(), letter))
    for match in re.finditer(r"\b([A-Z])\s+is\s+(?:the\s+)?correct\b", text, re.IGNORECASE):
        candidates.append((match.start(), match.group(1).upper()))
    for match in re.finditer(r"\\boxed\s*\{", text):
        letter = _option_at_start(text[match.start():])
        if letter is not None:
            candidates.append((match.start(), letter))
    if candidates:
        letter = max(candidates, key=lambda item: item[0])[1]
        return letter if letter in valid_letters else None

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and re.fullmatch(r"\(?[A-Z]\)?[.。]?", lines[-1], re.IGNORECASE):
        letter = _option_at_start(lines[-1])
        return letter if letter in valid_letters else None
    return None


def compute_gpqa_reward(response: str, label, metadata: dict | None = None) -> float:
    """Score a selected option; an unparsed response receives zero.

    Full answer text is accepted only as an exact normalized final answer,
    never by looking for the correct label or choice inside the response.
    """
    if response is None:
        return 0.0

    metadata = dict(metadata or {})

    choices = metadata.get("choices")
    if isinstance(choices, dict):
        choices = list(choices.values())
    elif choices is not None:
        choices = list(choices)

    valid_letters = metadata.get("valid_letters")
    if valid_letters:
        valid_letters = [str(letter).upper() for letter in valid_letters]
    elif choices:
        valid_letters = list(string.ascii_uppercase[: len(choices)])
    else:
        valid_letters = DEFAULT_VALID_LETTERS

    correct_letter = metadata.get("correct_letter")
    if isinstance(correct_letter, str):
        correct_letter = correct_letter.strip().upper()
    else:
        correct_letter = None

    label_text = None
    if isinstance(label, str):
        label_text = label.strip()
        if len(label_text) == 1 and label_text.upper() in valid_letters and not correct_letter:
            correct_letter = label_text.upper()
    elif isinstance(label, (int, float)):
        idx = int(label)
        if 0 <= idx < len(valid_letters):
            correct_letter = valid_letters[idx]

    if not correct_letter and choices and label_text:
        normalized_label = _normalize_text(label_text)
        for idx, choice in enumerate(choices):
            if _normalize_text(str(choice)) == normalized_label:
                correct_letter = valid_letters[idx]
                metadata.setdefault("correct_answer", choice)
                break

    extracted_letter = _extract_letter_from_response(response, valid_letters)
    if extracted_letter and correct_letter:
        return 1.0 if extracted_letter == correct_letter else 0.0

    candidate_answers = []
    if correct_letter and choices:
        try:
            idx = valid_letters.index(correct_letter)
        except ValueError:
            idx = None
        if idx is not None and idx < len(choices):
            candidate_answers.append(str(choices[idx]))

    for key in ("correct_answer", "answer_text"):
        value = metadata.get(key)
        if value:
            candidate_answers.append(str(value))

    if label_text and not (len(label_text) == 1 and label_text.upper() in valid_letters):
        candidate_answers.append(label_text)

    normalized_targets = {_normalize_text(text) for text in candidate_answers if text}
    text = _answer_text(response)
    markers = list(_FINAL_ANSWER.finditer(text)) or list(_NAMED_ANSWER.finditer(text))
    if markers:
        text = text[markers[-1].end():].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # An explicit answer starts immediately after its marker. Otherwise only
    # the last nonempty line can be a text answer.
    answer = (lines[0] if markers else lines[-1]) if lines else ""
    normalized_answer = _normalize_text(answer)
    if normalized_answer and normalized_answer in normalized_targets:
        if choices:
            matches = [choice for choice in choices if _normalize_text(str(choice)) == normalized_answer]
            if len(matches) != 1:
                return 0.0
        return 1.0

    if extracted_letter and not correct_letter and label_text:
        return 1.0 if extracted_letter == label_text.strip().upper() else 0.0

    return 0.0
