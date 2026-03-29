# qwen3-subtitles-pipeline

A subtitle pipeline using Qwen3 ASR with one main entry script:

- `make_subtitle.py`

Pipeline stages:

1. Silero VAD for compute-time chunking
2. Qwen3-ASR transcription
3. Optional Qwen3-ForcedAligner timing
4. Subtitle rechunking

## Requirements

- Python 3.12+
- `ffmpeg`
- `uv`

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install qwen-asr silero-vad soundfile
```

## Usage

### Basic

```bash
python make_subtitle.py input.mp4 \
  --language Japanese \
  --output input.ja.srt
```

### Skip alignment

```bash
python make_subtitle.py input.mp4 \
  --language Japanese \
  --no-align \
  --output input.ja.srt
```

### Key CLI flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--output` | `<input>.srt` | Output SRT path |
| `--language` | `Chinese` | Source language for ASR |
| `--no-align` | off | Skip forced alignment |
| `--device` | `cpu` | Torch device |
| `--dtype` | `float32` | Torch dtype (`float32`, `float16`, `bfloat16`) |
| `--max-sub-chars` | `22` | Max characters per subtitle cue |
| `--max-sub-duration` | `5.5` | Max seconds per subtitle cue |
| `--compute-max-chunk-s` | `90.0` | Soft ceiling for VAD chunk length |
| `--keep-temp` | off | Retain chunk WAVs in `--workdir` |

## Development

### Type checking

This project uses [basedpyright](https://github.com/DetachHead/basedpyright) for static type analysis.

```bash
uv run basedpyright make_subtitle.py
```

Several upstream dependencies (`soundfile`, `silero_vad`, `qwen_asr`) ship without type stubs. To suppress those warnings locally, uncomment the `[tool.basedpyright]` section in `pyproject.toml` and add the relevant libraries to `allowedUntypedLibraries`.

## Notes

- VAD and subtitle rechunking are handled in the same script.
- Use `--no-align` for faster runs when per-token timing is not needed.
