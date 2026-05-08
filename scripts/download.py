#!/usr/bin/env python3
"""Download a video via yt-dlp, or resolve a local file path.

Also fetches subtitles (manual first, then auto-generated) in VTT format so
transcribe.py can parse them without needing Whisper.

Google Drive URLs route through `gdrive.py` (gws CLI) when `gws` is available
on PATH; otherwise we fall back to yt-dlp for the public-file case.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import gdrive


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".flv", ".wmv"}
SUBTITLE_EXTS = {".vtt", ".srt"}


def is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https")


def _srt_to_vtt(srt_text: str) -> str:
    """Minimal SRT→VTT conversion: prepend WEBVTT header, swap `,` for `.` in cue timings."""
    lines = srt_text.splitlines()
    out: list[str] = ["WEBVTT", ""]
    for line in lines:
        if "-->" in line:
            out.append(line.replace(",", "."))
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def _materialize_sidecar(src: Path, work_dir: Path) -> Path:
    """Return a VTT file path. If src is already VTT, return it unchanged.
    If src is SRT, write a converted copy into work_dir and return that."""
    if src.suffix.lower() == ".vtt":
        return src
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "sidecar.vtt"
    out.write_text(_srt_to_vtt(src.read_text(encoding="utf-8", errors="replace")))
    return out


def _find_local_sidecar(video_path: Path) -> Path | None:
    """Look for a transcript file next to a video.

    Match rules (in order):
      1. Single .vtt/.srt file in the same folder — use it.
      2. Same-stem match (video.mp4 ↔ video.vtt) — use it.

    Single-file is preferred so patterns like Zoom's `<name>.mp4` +
    `<name>.transcript.vtt` (different stems) get caught for free.
    """
    folder = video_path.parent
    candidates = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUBTITLE_EXTS
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Multiple — fall back to strict same-stem match.
    stem = video_path.stem
    for c in candidates:
        if c.stem == stem:
            return c
    return None


def resolve_local(path: str, work_dir: Path | None = None) -> dict:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    if p.suffix.lower() not in VIDEO_EXTS:
        print(
            f"[watch] warning: {p.suffix} is not a known video extension, proceeding anyway",
            file=sys.stderr,
        )

    subtitle_path: str | None = None
    sidecar = _find_local_sidecar(p)
    if sidecar is not None:
        if work_dir is None:
            work_dir = p.parent
        materialized = _materialize_sidecar(sidecar, work_dir)
        subtitle_path = str(materialized)
        print(f"[watch] using sidecar transcript: {sidecar.name}", file=sys.stderr)

    return {
        "video_path": str(p),
        "subtitle_path": subtitle_path,
        "info": {"title": p.name, "url": str(p)},
        "downloaded": False,
    }


def _pick_subtitle(out_dir: Path) -> Path | None:
    candidates = sorted(out_dir.glob("video*.vtt"))
    if not candidates:
        return None
    preferred = [c for c in candidates if ".en" in c.name]
    return preferred[0] if preferred else candidates[0]


def _pick_video(out_dir: Path) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov"):
        for candidate in out_dir.glob(f"video*{ext}"):
            return candidate
    for candidate in out_dir.glob("video.*"):
        if candidate.suffix.lower() in VIDEO_EXTS:
            return candidate
    return None


def _ext_for_mime(mime: str | None, fallback_name: str) -> str:
    """Pick a file suffix from a Drive mimeType, falling back to the file name."""
    if mime:
        if mime == "video/mp4":
            return ".mp4"
        if mime == "video/quicktime":
            return ".mov"
        if mime == "video/x-matroska":
            return ".mkv"
        if mime == "video/webm":
            return ".webm"
    suffix = Path(fallback_name).suffix.lower()
    if suffix in VIDEO_EXTS:
        return suffix
    return ".mp4"


def _pick_drive_sidecar(subs: list[dict], video_name: str) -> dict | None:
    """Same match rules as _find_local_sidecar: single-file first, then same-stem."""
    if not subs:
        return None
    if len(subs) == 1:
        return subs[0]
    stem = Path(video_name).stem
    for s in subs:
        if Path(s.get("name") or "").stem == stem:
            return s
    return None


def _download_drive_sidecar(parent_id: str | None, video_name: str, out_dir: Path) -> Path | None:
    """If a transcript file lives next to the video on Drive, download + materialize it.
    Returns a VTT path or None."""
    if not parent_id:
        return None
    subs = gdrive.list_folder_subtitles(parent_id)
    chosen = _pick_drive_sidecar(subs, video_name)
    if chosen is None:
        return None

    sidecar_name = chosen.get("name") or "sidecar"
    suffix = Path(sidecar_name).suffix.lower() or ".vtt"
    raw_path = out_dir / f"sidecar{suffix}"
    print(f"[watch] using sidecar transcript: {sidecar_name}", file=sys.stderr)
    gdrive.download_file(chosen["id"], raw_path)
    return _materialize_sidecar(raw_path, out_dir)


def _gdrive_download_one(
    file_id: str,
    out_dir: Path,
    source_url: str,
    parent_id: str | None = None,
) -> dict:
    """Download a single Drive file via gws and shape the result like yt-dlp."""
    meta = gdrive.get_file_metadata(file_id)
    name = meta.get("name") or file_id
    mime = meta.get("mimeType") or ""
    if not mime.startswith("video/"):
        print(
            f"[watch] warning: Drive file mimeType is {mime!r}, expected video/*",
            file=sys.stderr,
        )

    out_path = out_dir / f"video{_ext_for_mime(mime, name)}"
    gdrive.download_file(file_id, out_path)

    if parent_id is None:
        parent_id = gdrive.get_file_parent(file_id)
    sidecar_path = _download_drive_sidecar(parent_id, name, out_dir)

    return {
        "video_path": str(out_path),
        "subtitle_path": str(sidecar_path) if sidecar_path else None,
        "info": {"title": name, "url": source_url},
        "downloaded": True,
    }


def _download_gdrive(url: str, classification: dict, out_dir: Path) -> dict | None:
    """Handle a Drive URL via gws. Returns None if gws is unavailable so caller
    can fall back to yt-dlp (only meaningful for the file case — folders need gws)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = classification["kind"]

    if kind == "file":
        if not gdrive.have_gws():
            print(
                "[watch] gws CLI not found — falling back to yt-dlp for this Drive file. "
                "For private files, install gws: https://github.com/googleworkspace/cli",
                file=sys.stderr,
            )
            return None
        return _gdrive_download_one(classification["id"], out_dir, url)

    if kind == "folder":
        if not gdrive.have_gws():
            raise SystemExit(
                "Drive folder URLs require the gws CLI "
                "(https://github.com/googleworkspace/cli). "
                "Either install gws, or pass a direct file URL like "
                "https://drive.google.com/file/d/<FILE_ID>/view"
            )
        videos = gdrive.list_folder_videos(classification["id"])
        chosen = gdrive.prompt_pick_video(videos)
        chosen_url = f"https://drive.google.com/file/d/{chosen['id']}/view"
        return _gdrive_download_one(
            chosen["id"], out_dir, chosen_url, parent_id=classification["id"]
        )

    return None


def download_url(url: str, out_dir: Path) -> dict:
    classification = gdrive.classify(url)
    if classification is not None:
        result = _download_gdrive(url, classification, out_dir)
        if result is not None:
            return result
        # gws unavailable for a file URL — fall through to yt-dlp.

    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp is not installed. Install with: brew install yt-dlp")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(out_dir / "video.%(ext)s")

    cmd = [
        "yt-dlp",
        "-N", "8",
        "-f", "bv*[height<=720]+ba/b[height<=720]/bv+ba/b",
        "--merge-output-format", "mp4",
        "--write-info-json",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs", "en,en-US,en-GB,en-orig",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--ignore-errors",
        "-o", output_template,
        url,
    ]

    # yt-dlp may exit non-zero if a subtitle variant fails (e.g. 429) even when
    # the video itself downloaded fine. Treat "video file present" as success.
    result = subprocess.run(cmd, stdout=sys.stderr, stderr=sys.stderr)
    video = _pick_video(out_dir)
    if video is None:
        raise SystemExit(
            f"yt-dlp did not produce a video file in {out_dir} (exit {result.returncode})"
        )

    subtitle = _pick_subtitle(out_dir)
    info_path = out_dir / "video.info.json"
    info: dict = {}
    if info_path.exists():
        try:
            raw = json.loads(info_path.read_text())
            info = {
                "title": raw.get("title"),
                "uploader": raw.get("uploader") or raw.get("channel"),
                "duration": raw.get("duration"),
                "url": raw.get("webpage_url") or url,
            }
        except Exception:
            info = {"url": url}

    return {
        "video_path": str(video),
        "subtitle_path": str(subtitle) if subtitle else None,
        "info": info or {"url": url},
        "downloaded": True,
    }


def download(source: str, out_dir: Path) -> dict:
    if is_url(source):
        return download_url(source, out_dir)
    return resolve_local(source, work_dir=out_dir)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: download.py <url-or-path> <out-dir>", file=sys.stderr)
        raise SystemExit(2)
    result = download(sys.argv[1], Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
