#!/usr/bin/env python3

import enum
import json
import re
import shutil
import subprocess
import sys
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, TypedDict, cast

import typer
from loguru import logger

import torch
import numpy as np

from qwen_asr import Qwen3ASRModel
from qwen_asr import Qwen3ForcedAligner
import soundfile as sf
from silero_vad import load_silero_vad, get_speech_timestamps


class DType(str, enum.Enum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"


class Translator(str, enum.Enum):
    deepseek = "deepseek"
    gemini = "gemini"


app = typer.Typer()


def run(cmd: list[str], check: bool = True) -> None:
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        if result.stdout:
            logger.error(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd)


def extract_audio(src: Path, dst: Path, sample_rate: int = 16000):
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(dst),
        ]
    )


def trim_audio(
    src: Path, dst: Path, start: float, end: float, sample_rate: int = 16000
):
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(dst),
        ]
    )


class SrtBlock(TypedDict):
    index: int
    timestamp: str
    text: str


class TranslationItem(TypedDict):
    index: int
    jp: str


@dataclass
class SpeechChunk:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TokenStamp:
    text: str
    start: float
    end: float


@dataclass
class SubtitleItem:
    start: float
    end: float
    text: str


class BreakStrength(enum.IntEnum):
    NONE = 0
    SOFT = 1
    MEDIUM = 2
    HARD = 3


@dataclass(frozen=True)
class BreakCandidate:
    pos: int
    strength: BreakStrength
    gap: float = 0.0


@dataclass(frozen=True)
class SegmentInfo:
    start_pos: int
    end_pos: int
    text: str
    start: float
    end: float
    raw_duration: float
    chars: int
    punct_chars: int


def detect_compute_chunks(
    audio_path: Path,
    threshold: float,
    min_silence_ms: int,
    min_speech_ms: int,
    speech_pad_ms: int,
    max_chunk_s: float | None = 90.0,
    hard_max_chunk_s: float | None = None,
) -> list[SpeechChunk]:
    model = load_silero_vad()  # pyright: ignore[reportUnknownVariableType]
    wav_np, sr = cast(tuple[np.ndarray, int], sf.read(str(audio_path), dtype="float32"))
    if wav_np.ndim > 1:
        wav_np = cast(np.ndarray, wav_np.mean(axis=1))
    if sr != 16000:
        raise RuntimeError(f"expected 16k audio after extraction, got {sr}")
    wav = torch.from_numpy(wav_np)  # pyright: ignore[reportUnknownMemberType]
    ts = cast(
        list[dict[str, float]],
        get_speech_timestamps(
            wav,
            model,
            sampling_rate=16000,
            threshold=threshold,
            min_silence_duration_ms=min_silence_ms,
            min_speech_duration_ms=min_speech_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=True,
        ),
    )
    if not ts:
        return []

    chunks: list[SpeechChunk] = []
    cur_start = float(ts[0]["start"])
    cur_end = float(ts[0]["end"])
    for item in ts[1:]:
        seg_start = float(item["start"])
        seg_end = float(item["end"])
        proposed_duration = seg_end - cur_start
        if max_chunk_s is not None and proposed_duration > max_chunk_s:
            chunks.append(SpeechChunk(cur_start, cur_end))
            cur_start, cur_end = seg_start, seg_end
        else:
            cur_end = seg_end
    chunks.append(SpeechChunk(cur_start, cur_end))

    if hard_max_chunk_s is None:
        return chunks

    split_chunks: list[SpeechChunk] = []
    for c in chunks:
        if c.duration <= hard_max_chunk_s:
            split_chunks.append(c)
            continue
        start = c.start
        while start < c.end:
            end = min(start + hard_max_chunk_s, c.end)
            split_chunks.append(SpeechChunk(start, end))
            start = end
    return split_chunks


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_sensevoice_text(text: str) -> str:
    for tag in [
        "<|zh|>",
        "<|en|>",
        "<|ja|>",
        "<|ko|>",
        "<|yue|>",
        "<|NEUTRAL|>",
        "<|HAPPY|>",
        "<|SAD|>",
        "<|ANGRY|>",
        "<|Speech|>",
        "<|woitn|>",
    ]:
        text = text.replace(tag, "")
    return normalize_text(text)


def init_qwen_asr(model_id: str, device: str, dtype_name: str) -> Qwen3ASRModel:
    return Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=getattr(torch, dtype_name),  # pyright: ignore [reportAny]
        max_inference_batch_size=1,
        max_new_tokens=2048,
        device_map=device,
    )


def init_aligner(model_id: str, device: str, dtype_name: str) -> Qwen3ForcedAligner:
    return Qwen3ForcedAligner.from_pretrained(
        model_id,
        device_map=device,
        dtype=getattr(torch, dtype_name),  # pyright: ignore [reportAny]
    )


def transcribe_chunk(
    model: Qwen3ASRModel, audio_path: Path, language: str | None
) -> str:
    results = model.transcribe(audio=str(audio_path), language=language)
    return normalize_text(results[0].text)


def force_align_chunk(
    aligner: Qwen3ForcedAligner, audio_path: Path, text: str, language: str
) -> list[TokenStamp]:
    aligned = aligner.align(audio=str(audio_path), text=text, language=language)[0]
    out: list[TokenStamp] = []
    for tok in aligned:
        out.append(
            TokenStamp(
                text=tok.text,
                start=tok.start_time,
                end=tok.end_time,
            )
        )
    return out


def inject_punctuation_tokens(
    original_text: str, tokens: list[TokenStamp]
) -> list[TokenStamp]:
    """Insert minimal-duration tokens for any characters dropped by the aligner.

    Treat the aligned token text as a subsequence of the original transcript:
    whenever the next character in ``original_text`` does not match the next
    aligned character, inject it back as a synthetic token anchored to the
    previous token boundary. Synthetic tokens get a tiny duration, capped by
    the next real token start so they never overlap with aligned speech.
    """
    if not tokens:
        return []

    aligned_chars: list[tuple[str, int]] = []
    for i, tok in enumerate(tokens):
        for ch in tok.text:
            aligned_chars.append((ch, i))

    aligned_pos = 0
    last_token_idx = -1
    result: list[TokenStamp] = []
    synthetic_duration_s = 0.01

    def append_synthetic(ch: str, next_token_idx: int | None) -> None:
        anchor = result[-1].end if result else tokens[0].start
        end = anchor + synthetic_duration_s
        if next_token_idx is not None:
            end = max(anchor, min(end, tokens[next_token_idx].start))
        result.append(TokenStamp(text=ch, start=anchor, end=end))

    for ch in original_text:
        if aligned_pos >= len(aligned_chars):
            append_synthetic(ch, None)
            continue

        aligned_ch, tok_idx = aligned_chars[aligned_pos]
        if ch != aligned_ch:
            append_synthetic(ch, tok_idx)
            continue

        if tok_idx != last_token_idx:
            result.append(tokens[tok_idx])
            last_token_idx = tok_idx
        aligned_pos += 1

    # Append any remaining tokens not yet covered
    for i in range(last_token_idx + 1, len(tokens)):
        result.append(tokens[i])

    return result


def contains_cjk(s: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in s)


def split_long_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    cur = ""
    for ch in text:
        if len(cur) >= max_chars:
            parts.append(cur)
            cur = ""
        cur += ch
    if cur:
        parts.append(cur)
    return parts


HARD_BREAK_PUNCT = set("。！？!?")
MEDIUM_BREAK_PUNCT = set("；;：:")
SOFT_BREAK_PUNCT = set("，、,")
TRAILING_CLOSERS = set("”’」』）》〉）)]】】")


def break_strength_for_char(ch: str) -> BreakStrength:
    if ch in HARD_BREAK_PUNCT:
        return BreakStrength.HARD
    if ch in MEDIUM_BREAK_PUNCT:
        return BreakStrength.MEDIUM
    if ch in SOFT_BREAK_PUNCT:
        return BreakStrength.SOFT
    return BreakStrength.NONE


def token_break_strength(text: str) -> BreakStrength:
    strength = BreakStrength.NONE
    for ch in text:
        strength = max(strength, break_strength_for_char(ch))
    return strength


def is_trailing_closer_token(text: str) -> bool:
    if not text:
        return False
    saw_closer = False
    for ch in text:
        if ch.isspace():
            continue
        if ch not in TRAILING_CLOSERS:
            return False
        saw_closer = True
    return saw_closer


def boundary_punctuation_strength(tokens: list[TokenStamp], boundary_pos: int) -> BreakStrength:
    k = boundary_pos - 1
    while k >= 0 and tokens[k].text.isspace():
        k -= 1
    saw_closer = False
    while k >= 0 and is_trailing_closer_token(tokens[k].text):
        saw_closer = True
        k -= 1
    if k < 0:
        return BreakStrength.NONE
    strength = token_break_strength(tokens[k].text)
    if strength == BreakStrength.NONE:
        return BreakStrength.NONE
    if saw_closer or k == boundary_pos - 1:
        return strength
    return BreakStrength.NONE


def boundary_pause_strength(
    tokens: list[TokenStamp], boundary_pos: int, silence_gap_s: float
) -> tuple[BreakStrength, float]:
    if boundary_pos <= 0 or boundary_pos >= len(tokens):
        return BreakStrength.NONE, 0.0
    gap = max(0.0, tokens[boundary_pos].start - tokens[boundary_pos - 1].end)
    if gap < silence_gap_s:
        return BreakStrength.NONE, gap
    if gap >= max(silence_gap_s * 2.0, silence_gap_s + 0.45):
        return BreakStrength.HARD, gap
    if gap >= max(silence_gap_s * 1.45, silence_gap_s + 0.2):
        return BreakStrength.MEDIUM, gap
    return BreakStrength.SOFT, gap


def build_break_candidates(tokens: list[TokenStamp], silence_gap_s: float) -> list[BreakCandidate]:
    boundary_map: dict[int, BreakCandidate] = {
        0: BreakCandidate(pos=0, strength=BreakStrength.HARD, gap=0.0)
    }
    for boundary_pos in range(1, len(tokens)):
        punct_strength = boundary_punctuation_strength(tokens, boundary_pos)
        pause_strength, gap = boundary_pause_strength(tokens, boundary_pos, silence_gap_s)
        strength = max(punct_strength, pause_strength)
        if strength == BreakStrength.NONE:
            continue
        prev = boundary_map.get(boundary_pos)
        candidate = BreakCandidate(pos=boundary_pos, strength=strength, gap=gap)
        if prev is None or candidate.strength > prev.strength or candidate.gap > prev.gap:
            boundary_map[boundary_pos] = candidate
    boundary_map[len(tokens)] = BreakCandidate(
        pos=len(tokens), strength=BreakStrength.HARD, gap=0.0
    )
    return [boundary_map[pos] for pos in sorted(boundary_map)]


def build_segment_info(
    tokens: list[TokenStamp],
    start_pos: int,
    end_pos: int,
    min_duration: float,
) -> SegmentInfo | None:
    seg_tokens = tokens[start_pos:end_pos]
    if not seg_tokens:
        return None

    text = "".join(t.text for t in seg_tokens).strip()
    if not text:
        return None

    start_tok = next((tok for tok in seg_tokens if tok.text.strip()), None)
    end_tok = next((tok for tok in reversed(seg_tokens) if tok.text.strip()), None)
    if start_tok is None or end_tok is None:
        return None

    raw_duration = max(0.0, end_tok.end - start_tok.start)
    punct_chars = 0
    chars = 0
    for ch in text:
        if ch.isspace():
            continue
        chars += 1
        if unicodedata.category(ch).startswith("P"):
            punct_chars += 1

    return SegmentInfo(
        start_pos=start_pos,
        end_pos=end_pos,
        text=text,
        start=start_tok.start,
        end=max(end_tok.end, start_tok.start + min_duration),
        raw_duration=raw_duration,
        chars=chars,
        punct_chars=punct_chars,
    )


def cube_badness(amount: float, scale: float) -> float:
    if amount <= 0:
        return 0.0
    return (amount / max(scale, 1e-6)) ** 3


def segment_badness(
    segment: SegmentInfo,
    end_boundary: BreakCandidate,
    skipped_soft: int,
    skipped_medium: int,
    skipped_hard: int,
    max_chars: int,
    max_duration: float,
    min_duration: float,
) -> float:
    if skipped_hard > 0:
        return float("inf")

    target_chars = max(8, round(max_chars * 0.82))
    min_chars = max(5, round(max_chars * 0.45))
    target_duration = min(max_duration * 0.72, 3.2)

    char_cost = 0.0
    if segment.chars < target_chars:
        char_cost += 40.0 * cube_badness(target_chars - segment.chars, max(target_chars - min_chars, 2))
    else:
        char_cost += 55.0 * cube_badness(segment.chars - target_chars, max(max_chars - target_chars, 2))
    if segment.chars < min_chars:
        char_cost += 140.0 * cube_badness(min_chars - segment.chars, max(min_chars, 2))
    if segment.chars > max_chars:
        char_cost += 260.0 * cube_badness(segment.chars - max_chars, max(max_chars, 2))

    duration_cost = 0.0
    if segment.raw_duration < min_duration:
        duration_cost += 18.0 * cube_badness(min_duration - segment.raw_duration, max(min_duration, 0.4))
    if segment.raw_duration > target_duration:
        duration_cost += 28.0 * cube_badness(
            segment.raw_duration - target_duration,
            max(max_duration - target_duration, 0.5),
        )
    if segment.raw_duration > max_duration:
        duration_cost += 120.0 * cube_badness(
            segment.raw_duration - max_duration,
            max(max_duration, 0.5),
        )

    density_allowance = 1 + segment.chars // 12
    punct_cost = 16.0 * cube_badness(
        segment.punct_chars - density_allowance,
        max(density_allowance, 1),
    )

    skip_cost = skipped_soft * 14.0 + skipped_medium * 48.0
    skip_cost += 6.0 * (skipped_soft + skipped_medium) ** 2

    end_cost = {
        BreakStrength.HARD: 0.0,
        BreakStrength.MEDIUM: 4.0,
        BreakStrength.SOFT: 10.0,
        BreakStrength.NONE: 18.0,
    }[end_boundary.strength]
    if end_boundary.gap > 0:
        end_cost -= min(end_boundary.gap, 1.2) * 6.0

    return char_cost + duration_cost + punct_cost + skip_cost + end_cost


def merge_close(
    items: list[SubtitleItem],
    gap_s: float = 0.12,
    max_chars: int = 24,
    max_duration: float = 6.5,
) -> list[SubtitleItem]:
    if not items:
        return []
    merged = [items[0]]
    for item in items[1:]:
        prev = merged[-1]
        prev_end_char = prev.text[-1] if prev.text else ""
        if (
            item.start - prev.end <= gap_s
            and len(prev.text + item.text) <= max_chars
            and (item.end - prev.start) <= max_duration
            and prev_end_char not in "。！？!?；;"
        ):
            prev.end = item.end
            prev.text += item.text
        else:
            merged.append(item)
    return merged


def rechunk_tokens(
    tokens: list[TokenStamp],
    max_chars: int,
    max_duration: float,
    min_duration: float = 0.9,
    silence_gap_s: float = 0.55,
) -> list[SubtitleItem]:
    if not tokens:
        return []
    candidates = build_break_candidates(tokens, silence_gap_s)
    if len(candidates) <= 2:
        text = "".join(t.text for t in tokens).strip()
        if not text:
            return []
        start = next((tok.start for tok in tokens if tok.text.strip()), tokens[0].start)
        end = next((tok.end for tok in reversed(tokens) if tok.text.strip()), tokens[-1].end)
        return [SubtitleItem(start, max(end, start + min_duration), text)]

    soft_prefix = [0]
    medium_prefix = [0]
    hard_prefix = [0]
    for candidate in candidates:
        soft_prefix.append(soft_prefix[-1] + int(candidate.strength == BreakStrength.SOFT))
        medium_prefix.append(
            medium_prefix[-1] + int(candidate.strength == BreakStrength.MEDIUM)
        )
        hard_prefix.append(hard_prefix[-1] + int(candidate.strength == BreakStrength.HARD))

    @lru_cache(maxsize=None)
    def get_segment(candidate_i: int, candidate_j: int) -> SegmentInfo | None:
        return build_segment_info(
            tokens,
            candidates[candidate_i].pos,
            candidates[candidate_j].pos,
            min_duration,
        )

    def skipped_counts(candidate_i: int, candidate_j: int) -> tuple[int, int, int]:
        lo = candidate_i + 1
        hi = candidate_j
        return (
            soft_prefix[hi] - soft_prefix[lo],
            medium_prefix[hi] - medium_prefix[lo],
            hard_prefix[hi] - hard_prefix[lo],
        )

    inf = float("inf")
    n = len(candidates)
    dp = [inf] * n
    prev = [-1] * n
    dp[0] = 0.0

    for end_idx in range(1, n):
        end_boundary = candidates[end_idx]
        best_cost = inf
        best_prev = -1

        for start_idx in range(end_idx - 1, -1, -1):
            if dp[start_idx] == inf:
                continue

            segment = get_segment(start_idx, end_idx)
            if segment is None:
                continue

            skipped_soft, skipped_medium, skipped_hard = skipped_counts(start_idx, end_idx)
            penalty = segment_badness(
                segment,
                end_boundary,
                skipped_soft,
                skipped_medium,
                skipped_hard,
                max_chars=max_chars,
                max_duration=max_duration,
                min_duration=min_duration,
            )
            if penalty == inf:
                continue

            total_cost = dp[start_idx] + penalty
            if total_cost < best_cost:
                best_cost = total_cost
                best_prev = start_idx

            if (
                skipped_hard > 0
                or segment.chars > max_chars * 2
                or segment.raw_duration > max_duration * 2
            ):
                break

        if best_prev >= 0:
            dp[end_idx] = best_cost
            prev[end_idx] = best_prev

    if prev[-1] < 0:
        text = "".join(t.text for t in tokens).strip()
        if not text:
            return []
        start = next((tok.start for tok in tokens if tok.text.strip()), tokens[0].start)
        end = next((tok.end for tok in reversed(tokens) if tok.text.strip()), tokens[-1].end)
        return [SubtitleItem(start, max(end, start + min_duration), text)]

    path: list[int] = []
    cur = n - 1
    while cur > 0:
        path.append(cur)
        cur = prev[cur]
        if cur < 0:
            break
    path.append(0)
    path.reverse()

    items: list[SubtitleItem] = []
    for start_idx, end_idx in zip(path, path[1:]):
        segment = get_segment(start_idx, end_idx)
        if segment is None:
            continue
        items.append(SubtitleItem(segment.start, segment.end, segment.text))
    return items


def srt_ts(sec: float) -> str:
    sec = max(sec, 0.0)
    ms = round(sec * 1000)
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def extend_subtitles(
    items: list[SubtitleItem], extend_s: float, min_gap_s: float = 0.08
) -> list[SubtitleItem]:
    """Extend each subtitle's end time, ensuring min_gap_s between consecutive cues."""
    if extend_s <= 0:
        return items
    for i, item in enumerate(items):
        max_end = items[i + 1].start - min_gap_s if i + 1 < len(items) else float("inf")
        item.end = max(item.start, min(item.end + extend_s, max_end))
    return items


class Stage(str, enum.Enum):
    EXTRACT = "extract"
    VAD = "vad"
    ASR = "asr"
    ALIGN = "align"
    RECHUNK = "rechunk"


def build_chunk_infos(chunks: list[SpeechChunk], chunks_dir: Path) -> list[tuple[int, SpeechChunk, Path]]:
    infos: list[tuple[int, SpeechChunk, Path]] = []
    for idx, chunk in enumerate(chunks, 1):
        infos.append((idx, chunk, chunks_dir / f"chunk_{idx:04d}.wav"))
    return infos


def run_vad(
    audio: Path,
    chunks_dir: Path,
    chunks_json: Path,
    threshold: float,
    min_silence_ms: int,
    min_speech_ms: int,
    speech_pad_ms: int,
    compute_max_chunk_s: float | None,
    hard_max_chunk_s: float | None,
) -> list[SpeechChunk]:
    logger.info(
        "[vad] threshold={}, min_silence_ms={}, min_speech_ms={}, max_chunk_s={}",
        threshold, min_silence_ms, min_speech_ms, compute_max_chunk_s,
    )
    chunks = detect_compute_chunks(
        audio, threshold, min_silence_ms, min_speech_ms, speech_pad_ms,
        compute_max_chunk_s, hard_max_chunk_s,
    )
    if not chunks:
        logger.error("No speech detected")
        sys.exit(2)
    logger.info("[vad] detected {} chunks", len(chunks))
    chunks_json.write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False),
        encoding="utf-8",
    )
    chunks_dir.mkdir(exist_ok=True)
    for idx, chunk in enumerate(chunks, 1):
        trim_audio(audio, chunks_dir / f"chunk_{idx:04d}.wav", chunk.start, chunk.end)
    return chunks


def load_chunks(chunks_json: Path, chunks_dir: Path) -> list[SpeechChunk]:
    if not chunks_json.exists():
        logger.error("chunks.json not found: {} (need --from-stage vad first)", chunks_json)
        sys.exit(2)
    data = json.loads(chunks_json.read_text(encoding="utf-8"))
    chunks = [SpeechChunk(**d) for d in data]
    for idx, chunk in enumerate(chunks, 1):
        chunk_path = chunks_dir / f"chunk_{idx:04d}.wav"
        if not chunk_path.exists():
            logger.error("chunk wav not found: {} (need --from-stage vad first)", chunk_path)
            sys.exit(2)
    logger.info("[vad] loaded {} chunks from {}", len(chunks), chunks_json)
    return chunks


def run_asr(
    chunk_infos: list[tuple[int, SpeechChunk, Path]],
    transcripts_json: Path,
    model: str,
    device: str,
    dtype: str,
    language: str,
) -> dict[int, str]:
    logger.info("[asr] loading engine: {} (device={}, dtype={})", model, device, dtype)
    engine = init_qwen_asr(model, device, dtype)
    transcripts: dict[int, str] = {}
    for idx, chunk, chunk_path in chunk_infos:
        logger.info("[asr] chunk {}/{} {:.2f}-{:.2f}s", idx, len(chunk_infos), chunk.start, chunk.end)
        try:
            transcripts[idx] = transcribe_chunk(engine, chunk_path, language)
        except Exception as e:
            logger.error("[asr] chunk {} failed: {}", idx, e)
    del engine
    import gc as _gc
    _ = _gc.collect()
    transcripts_json.write_text(
        json.dumps({str(k): v for k, v in transcripts.items()}, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("[asr] saved {} transcripts to {}", len(transcripts), transcripts_json)
    return transcripts


def load_transcripts(transcripts_json: Path) -> dict[int, str]:
    if not transcripts_json.exists():
        logger.error("transcripts.json not found: {} (need --from-stage asr first)", transcripts_json)
        sys.exit(2)
    raw = json.loads(transcripts_json.read_text(encoding="utf-8"))
    logger.info("[asr] loaded {} transcripts from {}", len(raw), transcripts_json)
    return {int(k): v for k, v in raw.items()}


def run_align(
    chunk_infos: list[tuple[int, SpeechChunk, Path]],
    transcripts: dict[int, str],
    tokens_json: Path,
    model: str,
    device: str,
    dtype: str,
    language: str,
    max_sub_chars: int,
    max_sub_duration: float,
    silence_gap_s: float,
) -> tuple[list[TokenStamp], list[SubtitleItem], list[dict[str, object]]]:
    logger.info("[align] loading engine: {} (device={}, dtype={})", model, device, dtype)
    aligner = init_aligner(model, device, dtype)
    all_tokens: list[TokenStamp] = []
    all_items: list[SubtitleItem] = []
    debug: list[dict[str, object]] = []

    for idx, chunk, chunk_path in chunk_infos:
        text = transcripts.get(idx, "").strip()
        logger.info("[align] chunk {}/{} {:.2f}-{:.2f}s", idx, len(chunk_infos), chunk.start, chunk.end)
        if not text:
            continue
        try:
            tokens = force_align_chunk(aligner, chunk_path, text, language)
            tokens = inject_punctuation_tokens(text, tokens)
            for t in tokens:
                t.start += chunk.start
                t.end += chunk.start
            all_tokens.extend(tokens)
        except Exception as e:
            logger.error("[align] chunk {} failed: {}", idx, e)
            continue
        subs = rechunk_tokens(tokens, max_sub_chars, max_sub_duration, silence_gap_s=silence_gap_s)
        for s in subs:
            all_items.append(s)
        debug.append({
            "chunk_index": idx,
            "start": chunk.start,
            "end": chunk.end,
            "transcript": text,
            "tokens": [asdict(t) for t in tokens],
            "subtitles": [asdict(s) for s in subs],
        })

    all_tokens.sort(key=lambda x: x.start)
    tokens_json.write_text(
        json.dumps([asdict(t) for t in all_tokens], ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("[align] saved {} tokens to {}", len(all_tokens), tokens_json)
    return all_tokens, all_items, debug


def run_rechunk_from_tokens(
    tokens_json: Path,
    max_sub_chars: int,
    max_sub_duration: float,
    silence_gap_s: float,
) -> list[SubtitleItem]:
    if not tokens_json.exists():
        logger.error("tokens.json not found: {} (need --from-stage align first)", tokens_json)
        sys.exit(2)
    data = json.loads(tokens_json.read_text(encoding="utf-8"))
    tokens = [TokenStamp(**d) for d in data]
    logger.info("[rechunk] loaded {} tokens from {}", len(tokens), tokens_json)
    return rechunk_tokens(tokens, max_sub_chars, max_sub_duration, silence_gap_s=silence_gap_s)


def run_no_align_rechunk(
    chunk_infos: list[tuple[int, SpeechChunk, Path]],
    transcripts: dict[int, str],
    max_sub_chars: int,
) -> tuple[list[SubtitleItem], list[dict[str, object]]]:
    logger.info("[rechunk] alignment skipped; coarse chunk timings + split")
    all_items: list[SubtitleItem] = []
    debug: list[dict[str, object]] = []
    for idx, chunk, _chunk_path in chunk_infos:
        text = transcripts.get(idx, "").strip()
        if not text:
            continue
        parts = split_long_text(text, max_sub_chars)
        span = max(chunk.end - chunk.start, 0.8)
        step = span / max(len(parts), 1)
        cur = chunk.start
        subs: list[SubtitleItem] = []
        for i, part in enumerate(parts):
            end = chunk.end if i == len(parts) - 1 else cur + step
            sub = SubtitleItem(cur, end, part)
            subs.append(sub)
            all_items.append(sub)
            cur = end
        debug.append({
            "chunk_index": idx,
            "start": chunk.start,
            "end": chunk.end,
            "transcript": text,
            "subtitles": [asdict(s) for s in subs],
        })
    return all_items, debug


def write_srt(items: list[SubtitleItem], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(items, 1):
            _ = f.write(
                f"{idx}\n{srt_ts(item.start)} --> {srt_ts(item.end)}\n{item.text}\n\n"
            )


def parse_srt_blocks(src: Path) -> list[SrtBlock]:
    text = src.read_text(encoding="utf-8").strip()
    blocks = [b for b in text.split("\n\n") if b.strip()]
    items: list[SrtBlock] = []
    for b in blocks:
        lines = b.splitlines()
        if len(lines) < 3:
            continue
        items.append(
            {
                "index": int(lines[0].strip()),
                "timestamp": lines[1],
                "text": " ".join(lines[2:]).strip(),
            }
        )
    return items


def write_bilingual_srt_from_map(src: Path, out: Path, translations: dict[int, str]):
    items = parse_srt_blocks(src)
    blocks: list[str] = []
    for item in items:
        zh = translations.get(item["index"], "").strip()
        blocks.append(f"{item['index']}\n{item['timestamp']}\n{item['text']}\n{zh}\n")
    _ = out.write_text("\n".join(blocks), encoding="utf-8")


@app.command()
def main(
    input: Annotated[Path, typer.Argument(help="Input media file")],
    output: Annotated[Path | None, typer.Option(help="Output SRT path")] = None,
    from_stage: Annotated[
        Stage, typer.Option("--from-stage", help="Start from this stage")
    ] = Stage.EXTRACT,
    workdir: Annotated[
        Path, typer.Option(help="Working directory for temp files")
    ] = Path("tmp/qwen3_subtitles"),  # pyright: ignore[reportCallInDefaultInitializer]
    language: Annotated[str, typer.Option(help="Source language for ASR")] = "Chinese",
    device: Annotated[str, typer.Option(help="Torch device")] = "cpu",
    dtype: Annotated[DType, typer.Option(help="Torch dtype")] = DType.float32,
    asr_model: Annotated[
        str, typer.Option(help="Qwen3 ASR model ID")
    ] = "Qwen/Qwen3-ASR-0.6B",
    aligner_model: Annotated[
        str, typer.Option(help="Qwen3 forced aligner model ID")
    ] = "Qwen/Qwen3-ForcedAligner-0.6B",
    no_align: Annotated[
        bool, typer.Option("--no-align", help="Skip forced alignment")
    ] = False,
    threshold: Annotated[float, typer.Option(help="VAD threshold")] = 0.6,
    min_silence_ms: Annotated[
        int, typer.Option(help="Min silence duration (ms)")
    ] = 600,
    min_speech_ms: Annotated[int, typer.Option(help="Min speech duration (ms)")] = 250,
    speech_pad_ms: Annotated[int, typer.Option(help="Speech padding (ms)")] = 100,
    compute_max_chunk_s: Annotated[
        float, typer.Option(help="Soft ceiling for VAD chunk length (s)")
    ] = 90.0,
    hard_max_chunk_s: Annotated[
        float | None, typer.Option(help="Hard max chunk length (s)")
    ] = None,
    max_sub_chars: Annotated[
        int, typer.Option(help="Max characters per subtitle cue")
    ] = 22,
    max_sub_duration: Annotated[
        float, typer.Option(help="Max seconds per subtitle cue")
    ] = 5.5,
    silence_gap_s: Annotated[
        float, typer.Option(help="Silence gap for rechunking (s)")
    ] = 0.55,
    sub_extend_s: Annotated[
        float, typer.Option(help="Extend subtitle end time by this many seconds")
    ] = 0.5,
    keep_temp: Annotated[
        bool, typer.Option("--keep-temp", help="Retain chunk WAVs")
    ] = False,
) -> None:
    """Integrated subtitle pipeline: VAD -> ASR backend -> optional align -> rechunk -> optional bilingual translation"""
    src = input.resolve()
    out = output.resolve() if output else src.with_suffix(".srt")
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    audio = workdir / f"{src.stem}.16k.wav"
    chunks_dir = workdir / "chunks"
    chunks_json = workdir / "chunks.json"
    transcripts_json = workdir / "transcripts.json"
    tokens_json = workdir / "tokens.json"

    # Stage: extract
    if from_stage <= Stage.EXTRACT:
        logger.info("[extract] {} -> {}", src, audio)
        extract_audio(src, audio)
    elif not audio.exists():
        logger.error("audio not found: {} (need --from-stage extract first)", audio)
        sys.exit(2)

    # Stage: vad
    if from_stage <= Stage.VAD:
        chunks = run_vad(
            audio, chunks_dir, chunks_json, threshold, min_silence_ms,
            min_speech_ms, speech_pad_ms, compute_max_chunk_s, hard_max_chunk_s,
        )
    else:
        chunks = load_chunks(chunks_json, chunks_dir)

    chunk_infos = build_chunk_infos(chunks, chunks_dir)

    # Stage: asr
    if from_stage <= Stage.ASR:
        transcripts = run_asr(
            chunk_infos, transcripts_json, asr_model, device, dtype, language,
        )
    else:
        transcripts = load_transcripts(transcripts_json)

    # Stage: align + rechunk
    if no_align:
        all_items, debug = run_no_align_rechunk(chunk_infos, transcripts, max_sub_chars)
    elif from_stage <= Stage.ALIGN:
        _tokens, all_items, debug = run_align(
            chunk_infos, transcripts, tokens_json, aligner_model, device, dtype,
            language, max_sub_chars, max_sub_duration, silence_gap_s,
        )
    else:
        all_items = run_rechunk_from_tokens(
            tokens_json, max_sub_chars, max_sub_duration, silence_gap_s,
        )
        debug = []

    # Write output
    all_items.sort(key=lambda x: x.start)
    if sub_extend_s > 0:
        extend_subtitles(all_items, sub_extend_s, min_gap_s=0.08)
        logger.info("extended subtitle end times by {:.2f}s (min gap 80ms)", sub_extend_s)
    logger.info("writing srt {}", out)
    write_srt(all_items, out)

    debug_path = workdir / ".debug.json"
    _ = debug_path.write_text(
        json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("debug file: {}", debug_path)

    if not keep_temp:
        shutil.rmtree(chunks_dir, ignore_errors=True)


if __name__ == "__main__":
    app()
