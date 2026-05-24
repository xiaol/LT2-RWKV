#!/usr/bin/env python3
"""Generate MiniMax narration and mux it into the LT2-RWKV Manim video."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[2]
ENV_PATH = Path("/home/xiaol/.codex/env/PaperX.env")
NARRATION = ROOT / "narration" / "lt2_rwkv_tts.md"
OUT_DIR = ROOT / "narration" / "audio" / "lt2_rwkv_minimax"
FINAL_WAV = OUT_DIR / "lt2_rwkv_minimax.wav"
FINAL_MP3 = OUT_DIR / "lt2_rwkv_minimax.mp3"
FINAL_SRT = ROOT / "narration" / "lt2_rwkv_minimax.srt"
SILENT_VIDEO = ROOT / "media/videos/lt2_rwkv_explainer/1080p29.97/LT2RWKVExplainer.mp4"
FINAL_VIDEO = ROOT / "outputs/LT2RWKVExplainer_minimax.mp4"
REPORT = ROOT / "analysis/lt2_rwkv_minimax_render_report.json"
FFMPEG = "/home/xiaol/.local/bin/ffmpeg"
FFPROBE = "/home/xiaol/.local/bin/ffprobe"

VOICE_A = "English_captivating_female1"
VOICE_B = "English_CaptivatingStoryteller"
MODEL = "speech-2.8-hd"
TAG_RE = re.compile(r"\((?:breath|chuckle|sighs|laughs)\)|<#[0-9.]+#>")


@dataclass(frozen=True)
class Turn:
    segment_index: int
    turn_index: int
    start: float
    end: float
    speaker: str
    text: str

    @property
    def voice(self) -> str:
        return VOICE_A if self.speaker == "A" else VOICE_B

    @property
    def subtitle_text(self) -> str:
        return re.sub(r"\s+", " ", TAG_RE.sub("", self.text)).strip()


@dataclass(frozen=True)
class Segment:
    index: int
    start: float
    end: float
    turns: list[Turn]

    @property
    def duration(self) -> float:
        return self.end - self.start


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def load_env() -> None:
    if not ENV_PATH.exists():
        return
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def minimax_url() -> str:
    base = os.environ.get("MINIMAX_TTS_BASE_URL") or os.environ.get("MINIMAX_BASE_URL") or "https://api.minimax.io/v1/t2a_v2"
    group_id = os.environ.get("MINIMAX_GROUP_ID")
    if group_id and "GroupId=" not in base:
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}GroupId={group_id}"
    return base


def parse_time(value: str) -> float:
    parts = value.strip().split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Bad timestamp: {value}")


def format_srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    whole = int(seconds)
    ms = int(round((seconds - whole) * 1000))
    if ms == 1000:
        whole += 1
        ms = 0
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_turns(body: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)(^|\s)([AB]):\s+", body))
    turns: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        speaker = match.group(2)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        text = re.sub(r"\s+", " ", body[start:end].strip())
        if text:
            turns.append((speaker, text))
    if not turns:
        turns.append(("A", re.sub(r"\s+", " ", body.strip())))
    return turns


def parse_segments() -> list[Segment]:
    text = NARRATION.read_text(encoding="utf-8")
    pattern = re.compile(r"^##\s+([0-9:.]+)-([0-9:.]+)\s*$", re.M)
    matches = list(pattern.finditer(text))
    segments: list[Segment] = []
    for seg_idx, match in enumerate(matches, start=1):
        start = parse_time(match.group(1))
        end = parse_time(match.group(2))
        body_start = match.end()
        body_end = matches[seg_idx].start() if seg_idx < len(matches) else len(text)
        parsed = parse_turns(text[body_start:body_end].strip())
        weights = [max(1, len(TAG_RE.sub("", turn_text).split())) for _, turn_text in parsed]
        total = sum(weights)
        cursor = start
        turns: list[Turn] = []
        for turn_idx, ((speaker, turn_text), _) in enumerate(zip(parsed, weights, strict=True), start=1):
            turn_end = end if turn_idx == len(parsed) else start + (end - start) * sum(weights[:turn_idx]) / total
            turns.append(Turn(seg_idx, turn_idx, cursor, turn_end, speaker, turn_text))
            cursor = turn_end
        segments.append(Segment(seg_idx, start, end, turns))
    return segments


def audio_duration(path: Path) -> float:
    result = run([
        FFPROBE,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ])
    return float(result.stdout.strip())


def cache_key(turn: Turn) -> str:
    payload = {"text": turn.text, "voice": turn.voice, "model": MODEL, "speed": 1.0, "version": 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def cached(path: Path, key: str) -> bool:
    meta = path.with_suffix(path.suffix + ".json")
    if not path.exists() or path.stat().st_size < 1000 or not meta.exists():
        return False
    try:
        return json.loads(meta.read_text(encoding="utf-8")).get("cache_key") == key
    except json.JSONDecodeError:
        return False


def synth_turn(turn: Turn) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / f"segment_{turn.segment_index:02d}_turn_{turn.turn_index:02d}_{turn.speaker}_raw.mp3"
    key = cache_key(turn)
    if cached(raw_path, key):
        return raw_path
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise RuntimeError("MINIMAX_API_KEY is not set")
    payload = {
        "model": MODEL,
        "text": turn.text,
        "stream": False,
        "language_boost": "English",
        "output_format": "hex",
        "voice_setting": {"voice_id": turn.voice, "speed": 1.0, "vol": 1.0, "pitch": 0},
        "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3", "channel": 1},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data: dict | None = None
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.post(minimax_url(), headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(3 * attempt)
    if data is None:
        raise RuntimeError(f"MiniMax TTS failed for segment {turn.segment_index} turn {turn.turn_index}: {last_error!r}")
    status = (data.get("base_resp") or {}).get("status_code", 0)
    if status not in (0, "0", None):
        raise RuntimeError(f"MiniMax TTS error for segment {turn.segment_index} turn {turn.turn_index}: {data.get('base_resp')}")
    audio_hex = (data.get("data") or {}).get("audio")
    audio_url = (data.get("data") or {}).get("audio_url")
    if audio_hex:
        raw_path.write_bytes(bytes.fromhex(audio_hex))
    elif audio_url:
        audio = requests.get(audio_url, timeout=180)
        audio.raise_for_status()
        raw_path.write_bytes(audio.content)
    else:
        raise RuntimeError(f"No audio in MiniMax response: {json.dumps(data)[:500]}")
    raw_path.with_suffix(raw_path.suffix + ".json").write_text(
        json.dumps({"cache_key": key, "provider": "MiniMax", "model": MODEL, "voice": turn.voice, "speaker": turn.speaker}, indent=2),
        encoding="utf-8",
    )
    return raw_path


def normalize_turn(turn: Turn, raw_path: Path) -> Path:
    norm = OUT_DIR / f"segment_{turn.segment_index:02d}_turn_{turn.turn_index:02d}_{turn.speaker}_norm.wav"
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_path), "-af", "loudnorm=I=-20:TP=-1.5:LRA=11", "-ar", "48000", "-ac", "2", str(norm)])
    return norm


def concat_files(paths: list[Path], out_path: Path) -> None:
    concat = out_path.with_suffix(".concat.txt")
    concat.write_text("".join(f"file '{p.resolve()}'\n" for p in paths), encoding="utf-8")
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-ar", "48000", "-ac", "2", str(out_path)])


def fit_segment(segment: Segment, turn_paths: list[Path]) -> tuple[Path, dict[str, float]]:
    segment_raw = OUT_DIR / f"segment_{segment.index:02d}_joined.wav"
    concat_files(turn_paths, segment_raw)
    raw_duration = audio_duration(segment_raw)
    target = max(0.1, segment.duration - 0.20)
    speed = 1.0
    filters: list[str] = []
    if raw_duration > target:
        speed = min(1.18, raw_duration / target)
        filters.append(f"atempo={speed:.5f}")
    filters.extend(["loudnorm=I=-20:TP=-1.5:LRA=11", "apad"])
    fitted = OUT_DIR / f"segment_{segment.index:02d}_fitted.wav"
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(segment_raw), "-af", ",".join(filters), "-t", f"{segment.duration:.3f}", "-ar", "48000", "-ac", "2", str(fitted)])
    return fitted, {"raw_duration": raw_duration, "target_duration": segment.duration, "speed_factor": speed}


def write_final_audio(segment_paths: list[Path], duration: float) -> None:
    concat_files(segment_paths, FINAL_WAV)
    tmp = FINAL_WAV.with_suffix(".tmp.wav")
    FINAL_WAV.rename(tmp)
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp), "-af", "loudnorm=I=-20:TP=-1.5:LRA=11,apad", "-t", f"{duration:.3f}", "-ar", "48000", "-ac", "2", str(FINAL_WAV)])
    tmp.unlink(missing_ok=True)
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(FINAL_WAV), "-c:a", "libmp3lame", "-b:a", "128k", str(FINAL_MP3)])


def split_subtitle_text(text: str, start: float, end: float, first_index: int) -> tuple[list[str], int]:
    chunks: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= 104:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    step = (end - start) / max(1, len(chunks))
    entries: list[str] = []
    index = first_index
    for i, chunk in enumerate(chunks):
        a = start + step * i
        b = end if i == len(chunks) - 1 else start + step * (i + 1)
        entries.append(f"{index}\n{format_srt_time(a)} --> {format_srt_time(b)}\n{chunk}\n\n")
        index += 1
    return entries, index


def write_srt(segments: list[Segment]) -> None:
    entries: list[str] = []
    index = 1
    for segment in segments:
        for turn in segment.turns:
            prefix = "A: " if turn.speaker == "A" else "B: "
            turn_entries, index = split_subtitle_text(prefix + turn.subtitle_text, turn.start, turn.end, index)
            entries.extend(turn_entries)
    FINAL_SRT.write_text("".join(entries).rstrip() + "\n", encoding="utf-8")


def mux_video() -> dict[str, object]:
    FINAL_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    run([
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(SILENT_VIDEO),
        "-i",
        str(FINAL_WAV),
        "-i",
        str(FINAL_SRT),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map",
        "2:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-c:s",
        "mov_text",
        "-metadata:s:s:0",
        "language=eng",
        "-shortest",
        str(FINAL_VIDEO),
    ])
    result = run([FFPROBE, "-v", "error", "-show_entries", "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate", "-of", "json", str(FINAL_VIDEO)])
    return json.loads(result.stdout)


def main() -> int:
    load_env()
    if not SILENT_VIDEO.exists():
        raise RuntimeError(f"Missing silent video: {SILENT_VIDEO}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    segments = parse_segments()
    video_duration = audio_duration(SILENT_VIDEO)
    fitted_segments: list[Path] = []
    segment_reports: list[dict[str, float | int]] = []
    for segment in segments:
        print(json.dumps({"segment": segment.index, "turns": len(segment.turns), "start": segment.start, "end": segment.end}, sort_keys=True))
        turn_paths = [normalize_turn(turn, synth_turn(turn)) for turn in segment.turns]
        fitted, metrics = fit_segment(segment, turn_paths)
        fitted_segments.append(fitted)
        segment_reports.append({"segment": segment.index, **metrics})
    write_final_audio(fitted_segments, video_duration)
    write_srt(segments)
    streams = mux_video()
    report = {
        "title": "LT2-RWKV short explainer",
        "tts": {
            "provider": "MiniMax",
            "model": MODEL,
            "voice_a": VOICE_A,
            "voice_b": VOICE_B,
            "speed": 1.0,
            "pitch": 0,
            "volume": 1.0,
            "normalization": "ffmpeg loudnorm=I=-20:TP=-1.5:LRA=11 per turn, per segment, and final mix",
        },
        "max_speed_factor": max(float(item["speed_factor"]) for item in segment_reports),
        "audio_duration": audio_duration(FINAL_WAV),
        "video": str(FINAL_VIDEO),
        "silent_video": str(SILENT_VIDEO),
        "wav": str(FINAL_WAV),
        "mp3": str(FINAL_MP3),
        "srt": str(FINAL_SRT),
        "streams": streams,
        "segment_reports": segment_reports,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
