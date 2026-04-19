#!/usr/bin/env python3

import enum
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
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
    hard_punct = set("。！？!?；;")
    soft_punct = set("，、：,:")
    items: list[SubtitleItem] = []
    buf: list[TokenStamp] = []

    def flush(force: bool = False):
        nonlocal buf
        if not buf:
            return
        raw_text = "".join(t.text for t in buf).strip()
        if not raw_text:
            buf = []
            return
        start = buf[0].start
        end = max(buf[-1].end, start + min_duration)
        if (not force) and len(raw_text) > max_chars * 1.6:
            subparts = split_long_text(raw_text, max_chars)
            span = max(end - start, min_duration)
            step = span / len(subparts)
            cur_start = start
            for i, part in enumerate(subparts):
                cur_end = end if i == len(subparts) - 1 else cur_start + step
                items.append(SubtitleItem(cur_start, cur_end, part))
                cur_start = cur_end
        else:
            items.append(SubtitleItem(start, end, raw_text))
        buf = []

    for tok in tokens:
        if not tok.text.strip():
            continue
        gap = max(0.0, tok.start - buf[-1].end) if buf else 0.0
        if buf and gap >= silence_gap_s:
            flush()
        buf.append(tok)
        text_now = "".join(t.text for t in buf).strip()
        duration_now = buf[-1].end - buf[0].start
        last_char = buf[-1].text[-1] if buf[-1].text else ""
        if (
            last_char in hard_punct
            or (last_char in soft_punct and len(text_now) >= max(8, max_chars // 2))
            or duration_now >= max_duration
            or (len(text_now) >= max_chars and contains_cjk(text_now))
        ):
            flush(force=True)
    flush(force=True)
    return merge_close(
        items, gap_s=0.12, max_chars=max_chars, max_duration=max_duration + 0.8
    )


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
        item.end = min(item.end + extend_s, max_end)
    return items


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
    asr_model_id = asr_model
    aligner_model_id = aligner_model

    workdir.mkdir(parents=True, exist_ok=True)
    audio = workdir / f"{src.stem}.16k.wav"
    chunks_dir = workdir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    logger.info("extracting audio: {} -> {}", src, audio)
    extract_audio(src, audio)

    logger.info(
        f"compute chunks via silero vad ({threshold=}, {min_silence_ms=}, {min_speech_ms=}, {compute_max_chunk_s=}, {hard_max_chunk_s=})"
    )
    chunks = detect_compute_chunks(
        audio,
        threshold,
        min_silence_ms,
        min_speech_ms,
        speech_pad_ms,
        compute_max_chunk_s,
        hard_max_chunk_s,
    )
    if not chunks:
        logger.error("No speech detected")
        sys.exit(2)
    logger.info("detected {} compute chunks", len(chunks))

    chunk_infos: list[tuple[int, SpeechChunk, Path]] = []
    for idx, chunk in enumerate(chunks, 1):
        chunk_path = chunks_dir / f"chunk_{idx:04d}.wav"
        trim_audio(audio, chunk_path, chunk.start, chunk.end)
        chunk_infos.append((idx, chunk, chunk_path))

    logger.info(
        "loading ASR engine: {} (device={}, dtype={})",
        asr_model_id,
        device,
        dtype,
    )
    asr = init_qwen_asr(asr_model_id, device, dtype)
    transcripts: dict[int, str] = {}
    for idx, chunk, chunk_path in chunk_infos:
        logger.info(
            "asr compute chunk {}/{} {:.2f}-{:.2f}s",
            idx,
            len(chunk_infos),
            chunk.start,
            chunk.end,
        )
        try:
            transcripts[idx] = transcribe_chunk(asr, chunk_path, language)
        except Exception as e:
            logger.error("asr chunk {} failed: {}", idx, e)
    del asr
    import gc

    _ = gc.collect()

    all_items: list[SubtitleItem] = []
    debug: list[dict[str, object]] = []

    if no_align:
        logger.info("alignment skipped; using coarse chunk timings + rechunk-by-text")
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
            debug.append(
                {
                    "chunk_index": idx,
                    "start": chunk.start,
                    "end": chunk.end,
                    "transcript": text,
                    "subtitles": [asdict(s) for s in subs],
                }
            )
    else:
        logger.info("loading qwen forced aligner...")
        aligner = init_aligner(aligner_model_id, device, dtype)
        for idx, chunk, chunk_path in chunk_infos:
            text = transcripts.get(idx, "").strip()
            logger.info(
                "align compute chunk {}/{} {:.2f}-{:.2f}s",
                idx,
                len(chunk_infos),
                chunk.start,
                chunk.end,
            )
            if not text:
                continue
            try:
                tokens = force_align_chunk(aligner, chunk_path, text, language)
            except Exception as e:
                logger.error("align chunk {} failed: {}", idx, e)
                continue
            subs = rechunk_tokens(
                tokens,
                max_sub_chars,
                max_sub_duration,
                silence_gap_s=silence_gap_s,
            )
            for s in subs:
                s.start += chunk.start
                s.end += chunk.start
                all_items.append(s)
            debug.append(
                {
                    "chunk_index": idx,
                    "start": chunk.start,
                    "end": chunk.end,
                    "transcript": text,
                    "tokens": [asdict(t) for t in tokens],
                    "subtitles": [asdict(s) for s in subs],
                }
            )

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
