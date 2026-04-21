"""Shared utilities for LLM API calls with JSON validation and feedback retry."""

import json
import re
from collections.abc import Set as AbstractSet

from loguru import logger
from openai import OpenAI


FEEDBACK_TEMPLATE = """\
Your previous response failed JSON validation.
Errors/Issues to fix: {issues}

Original input:
{original}

Instructions:
1. Fix the JSON syntax errors (e.g., check for unescaped quotes, trailing commas, or missing brackets).
2. Ensure the output contains exactly these keys: {keys}
3. Output ONLY a valid JSON object starting with `{{` and ending with `}}`. Do not include ANY text outside the JSON."""


def extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from LLM response (handles markdown fences)."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def validate_dict_string_values(result: dict, expected_keys: AbstractSet[str]) -> list[str]:
    """Return list of issue descriptions (empty == valid).

    Checks that *result* is a dict with exactly *expected_keys* and all values are strings.
    """
    issues: list[str] = []
    if not isinstance(result, dict):
        issues.append("Response is not a JSON object")
        return issues
    result_keys = set(result.keys())
    missing = expected_keys - result_keys
    extra = result_keys - expected_keys
    if missing:
        issues.append(f"Missing keys: {sorted(missing)}")
    if extra:
        issues.append(f"Extra keys: {sorted(extra)}")
    for k, v in result.items():
        if not isinstance(v, str):
            issues.append(f"Value for key '{k}' is not a string")
            break
    return issues


def llm_call_with_feedback(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_msg: str,
    expected_keys: AbstractSet[str],
    *,
    max_retries: int = 3,
    temperature: float = 0.3,
) -> dict:
    """Call the LLM with a retry + feedback loop.

    Sends *system_prompt* and *user_msg*, parses the response as JSON,
    validates it with :func:`validate_dict_string_values`, and on failure
    appends the assistant response plus :data:`FEEDBACK_TEMPLATE` to the
    message list before retrying.

    Returns the parsed dict on success.

    Raises
        ValueError: if all retries are exhausted without a valid response.
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    for attempt in range(1, max_retries + 2):  # 1 initial + max_retries feedback
        resp = client.chat.completions.create(
            model=model,
            messages=messages,  # pyright: ignore[reportArgumentType]
            temperature=temperature,
        )
        raw = resp.choices[0].message.content or ""
        parsed = extract_json(raw)

        if parsed is None:
            issues_text = "Response is not valid JSON"
            logger.warning("LLM response attempt {}: not valid JSON", attempt)
        else:
            issues = validate_dict_string_values(parsed, expected_keys)
            if not issues:
                return parsed
            issues_text = "; ".join(issues)
            logger.warning("LLM response attempt {}: {}", attempt, issues_text)

        if attempt > max_retries:
            raise ValueError(
                f"LLM call failed after {max_retries} retries: {issues_text}"
            )

        feedback = FEEDBACK_TEMPLATE.format(
            issues=issues_text,
            original=user_msg,
            keys=sorted(expected_keys),
        )
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": feedback})

    raise ValueError("Unexpected: loop exhausted without returning or raising")
