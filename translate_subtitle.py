#!/usr/bin/env python3
"""SRT subtitle translation via LLM (DeepSeek / OpenAI-compatible API).

Workflow:
  1. Parse SRT → {index: text} dict
  2. Split into batches (batch_size)
  3. Send each batch as JSON to LLM, receive translated JSON
  4. Validate response (keys match, no extras, correct type)
  5. Retry with feedback on validation failure
  6. Write bilingual SRT output
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from llm_utils import extract_json, llm_call_with_feedback

load_dotenv()

app = typer.Typer()

# ── SRT parsing ──────────────────────────────────────────────────────────────

_BLOCK_RE = re.compile(r"\r?\n\r?\n")


def parse_srt(path: Path) -> list[dict]:
    """Return list of {index, timestamp, text}."""
    raw = path.read_text(encoding="utf-8").strip()
    blocks = [b for b in _BLOCK_RE.split(raw) if b.strip()]
    items: list[dict] = []
    for b in blocks:
        lines = b.splitlines()
        if len(lines) < 3:
            continue
        items.append(
            {
                "index": int(lines[0].strip()),
                "timestamp": lines[1].strip(),
                "text": " ".join(lines[2:]).strip(),
            }
        )
    return items


def srt_to_dict(blocks: list[dict]) -> dict[str, str]:
    """Convert parsed SRT blocks to {"1": "text", "2": "text", ...}."""
    return {str(b["index"]): b["text"] for b in blocks}


# ── Batching ─────────────────────────────────────────────────────────────────


def make_batches(d: dict[str, str], batch_size: int) -> list[dict[str, str]]:
    keys = list(d.keys())
    return [
        {k: d[k] for k in keys[i : i + batch_size]}
        for i in range(0, len(keys), batch_size)
    ]


# ── LLM interaction ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a professional subtitle translator.
You will receive a JSON object where each key is a subtitle index and each value is the original subtitle text.
Translate every value into {target_lang}. Keep each key unchanged.

Translation Guidelines:
1. Contextual Flow: The keys represent sequential subtitles. Consider the context of adjacent lines to ensure the translation is coherent and natural.
2. Conciseness: Keep translations concise and suitable for screen reading.

Output Requirements:
- Return ONLY a valid JSON object.
- Start your response exactly with `{{` and end with `}}`.
- NO markdown fences (do NOT use ```json). NO explanations.
- CRITICAL: Properly escape any double quotes inside the translated strings (e.g., use \\" instead of ").

Example Input:
{{"1": "I said,", "2": "let it go!"}}

Example Output:
{{"1": "我说了，", "2": "随它去吧！"}}"""

def _build_system_prompt(target_lang: str) -> str:
    return SYSTEM_PROMPT.format(target_lang=target_lang)


def translate_batch(
    client: OpenAI,
    model: str,
    batch: dict[str, str],
    target_lang: str,
    max_retries: int = 3,
) -> dict[str, str]:
    """Translate one batch with validation + feedback retry."""
    expected_keys = set(batch.keys())
    system = _build_system_prompt(target_lang)
    user_msg = json.dumps(batch, ensure_ascii=False)
    try:
        result = llm_call_with_feedback(
            client, model, system, user_msg, expected_keys, max_retries=max_retries
        )
        return {k: str(v) for k, v in result.items()}
    except ValueError:
        logger.error("batch keys {}-{} failed after {} retries, returning originals", min(expected_keys), max(expected_keys), max_retries)
        return dict(batch)


# ── Output ───────────────────────────────────────────────────────────────────


def write_bilingual_srt(
    blocks: list[dict],
    translations: dict[str, str],
    out: Path,
    translation_first: bool = True,
) -> None:
    """Write SRT with two lines per cue. translation_first puts target lang on top."""
    parts: list[str] = []
    for b in blocks:
        idx = str(b["index"])
        translated = translations.get(idx, "").strip()
        if translation_first:
            line1, line2 = translated, b["text"]
        else:
            line1, line2 = b["text"], translated
        parts.append(f"{b['index']}\n{b['timestamp']}\n{line1}\n{line2}\n")
    out.write_text("\n".join(parts), encoding="utf-8")


def write_translated_srt(
    blocks: list[dict], translations: dict[str, str], out: Path
) -> None:
    """Write SRT with only translated text."""
    parts: list[str] = []
    for b in blocks:
        idx = str(b["index"])
        translated = translations.get(idx, b["text"]).strip()
        parts.append(f"{b['index']}\n{b['timestamp']}\n{translated}\n")
    out.write_text("\n".join(parts), encoding="utf-8")


# ── CLI ──────────────────────────────────────────────────────────────────────


@app.command()
def main(
    input: Annotated[Path, typer.Argument(help="Input SRT file")],
    output: Annotated[Path | None, typer.Option(help="Output SRT path")] = None,
    target_lang: Annotated[
        str, typer.Option(help="Target language for translation")
    ] = "简体中文",
    model: Annotated[
        str, typer.Option(help="LLM model name")
    ] = "deepseek-chat",
    batch_size: Annotated[
        int, typer.Option(help="Subtitles per LLM request")
    ] = 20,
    thread_num: Annotated[
        int, typer.Option(help="Parallel request threads")
    ] = 4,
    max_retries: Annotated[
        int, typer.Option(help="Max feedback retries per batch")
    ] = 3,
    bilingual: Annotated[
        bool, typer.Option("--bilingual", help="Output bilingual SRT (original + translated)")
    ] = True,
    translation_first: Annotated[
        bool,
        typer.Option(help="In bilingual mode, put translation on top (first line)"),
    ] = True,
    api_key: Annotated[
        str | None, typer.Option(envvar="LLM_API_KEY", help="API key (or set LLM_API_KEY)")
    ] = None,
    api_base: Annotated[
        str | None,
        typer.Option(envvar="LLM_API_BASE", help="API base URL (or set LLM_API_BASE)"),
    ] = None,
) -> None:
    """Translate an SRT subtitle file via LLM."""
    if not api_key:
        logger.error("API key required: pass --api-key or set LLM_API_KEY in .env")
        sys.exit(1)

    base_url = api_base or "https://api.deepseek.com"
    client = OpenAI(api_key=api_key, base_url=base_url)

    src = input.resolve()
    if not src.exists():
        logger.error("Input file not found: {}", src)
        sys.exit(1)

    logger.info("parsing SRT: {}", src)
    blocks = parse_srt(src)
    if not blocks:
        logger.error("No subtitle blocks found")
        sys.exit(1)
    logger.info("found {} subtitle blocks", len(blocks))

    subtitle_dict = srt_to_dict(blocks)
    batches = make_batches(subtitle_dict, batch_size)
    logger.info(
        "split into {} batches (batch_size={}, threads={})",
        len(batches),
        batch_size,
        thread_num,
    )

    all_translations: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=thread_num) as pool:
        futures = {
            pool.submit(
                translate_batch, client, model, batch, target_lang, max_retries
            ): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_idx = futures[future]
            try:
                result = future.result()
                all_translations.update(result)
                logger.info(
                    "batch {}/{} done ({} items)",
                    batch_idx + 1,
                    len(batches),
                    len(result),
                )
            except Exception as e:
                logger.error("batch {} failed: {}", batch_idx + 1, e)

    # determine output path
    if output:
        out = output.resolve()
    else:
        stem = src.stem
        out = src.with_name(f"{stem}.translated.srt")

    if bilingual:
        write_bilingual_srt(blocks, all_translations, out, translation_first)
    else:
        write_translated_srt(blocks, all_translations, out)

    logger.info("wrote {} ({} entries)", out, len(all_translations))


if __name__ == "__main__":
    app()
