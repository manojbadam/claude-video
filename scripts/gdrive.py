#!/usr/bin/env python3
"""Google Drive helpers for /watch.

`yt-dlp` handles public, non-interstitial Drive files inconsistently and can't
enumerate folders at all. This module uses the `gws` CLI
(https://github.com/googleworkspace/cli) when available to:

  - Download a Drive file by ID via the authenticated Drive API
  - List videos inside a Drive folder so the user can pick one

`gws` is optional: if it isn't on PATH, callers fall back to `yt-dlp`. Folder
URLs are the only case that strictly requires it.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse


_FILE_PATH_RE = re.compile(r"^/file/d/([A-Za-z0-9_-]+)")
_FOLDER_PATH_RE = re.compile(r"^/drive(?:/u/\d+)?/folders/([A-Za-z0-9_-]+)")


def classify(url: str) -> dict | None:
    """Return {kind: file|folder, id: str} for a Drive URL, else None."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.netloc != "drive.google.com":
        return None

    m = _FILE_PATH_RE.match(parsed.path)
    if m:
        return {"kind": "file", "id": m.group(1)}

    if parsed.path == "/open":
        qs = parse_qs(parsed.query)
        ids = qs.get("id")
        if ids:
            return {"kind": "file", "id": ids[0]}

    m = _FOLDER_PATH_RE.match(parsed.path)
    if m:
        return {"kind": "folder", "id": m.group(1)}

    return None


def have_gws() -> bool:
    return shutil.which("gws") is not None


def _run_gws(
    args: list[str],
    capture: bool = True,
    cwd: Path | str | None = None,
) -> subprocess.CompletedProcess:
    """Invoke `gws` and surface stderr verbatim on failure.

    `cwd` is required for `--output` paths under gws >= 0.22, which rejects
    output paths that resolve outside the current working directory.
    """
    return subprocess.run(
        ["gws", *args],
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(cwd) if cwd else None,
    )


def get_file_metadata(file_id: str) -> dict:
    """Return {id, name, mimeType, size?} for a Drive file ID. Raises SystemExit on failure."""
    params = json.dumps({
        "fileId": file_id,
        "fields": "id,name,mimeType,size",
        "supportsAllDrives": True,
    })
    proc = _run_gws(["drive", "files", "get", "--params", params])
    if proc.returncode != 0:
        raise SystemExit(
            f"gws drive files get failed for {file_id}: {proc.stderr.strip() or 'unknown error'}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"gws returned non-JSON metadata: {exc}: {proc.stdout[:200]}")


def download_file(file_id: str, out_path: Path) -> Path:
    """Download a Drive file by ID to `out_path`. Returns the path. Raises SystemExit."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = json.dumps({
        "fileId": file_id,
        "alt": "media",
        "supportsAllDrives": True,
    })
    print(f"[watch] downloading via gws → {out_path.name}…", file=sys.stderr)
    # gws >= 0.22 rejects --output paths outside the CWD as a security
    # validation. Run the subprocess inside the target directory and pass
    # only the file name.
    proc = _run_gws(
        [
            "drive", "files", "get",
            "--params", params,
            "--output", out_path.name,
        ],
        capture=True,
        cwd=out_path.parent,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"gws download failed: {proc.stderr.strip() or 'unknown error'}"
        )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise SystemExit(f"gws produced empty file at {out_path}")
    return out_path


def _list_folder(folder_id: str) -> list[dict]:
    """Return ALL non-trashed children of a Drive folder."""
    params = json.dumps({
        "q": f"'{folder_id}' in parents and trashed=false",
        "fields": "files(id,name,mimeType,size,parents)",
        "supportsAllDrives": True,
        "includeItemsFromAllDrives": True,
        "pageSize": 1000,
    })
    proc = _run_gws(["drive", "files", "list", "--params", params])
    if proc.returncode != 0:
        raise SystemExit(
            f"gws drive files list failed for folder {folder_id}: "
            f"{proc.stderr.strip() or 'unknown error'}"
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"gws returned non-JSON folder listing: {exc}: {proc.stdout[:200]}")
    return data.get("files") or []


def list_folder_videos(folder_id: str) -> list[dict]:
    """Return video files in a Drive folder."""
    return [f for f in _list_folder(folder_id) if (f.get("mimeType") or "").startswith("video/")]


def list_folder_subtitles(folder_id: str) -> list[dict]:
    """Return likely subtitle files (.vtt / .srt by name or mimeType) in a Drive folder."""
    out: list[dict] = []
    for f in _list_folder(folder_id):
        name = (f.get("name") or "").lower()
        mime = (f.get("mimeType") or "").lower()
        if name.endswith(".vtt") or name.endswith(".srt"):
            out.append(f)
            continue
        if mime in ("text/vtt", "application/x-subrip"):
            out.append(f)
    return out


def get_file_parent(file_id: str) -> str | None:
    """Return the first parent folder ID of a Drive file, or None if unknown."""
    params = json.dumps({
        "fileId": file_id,
        "fields": "parents",
        "supportsAllDrives": True,
    })
    proc = _run_gws(["drive", "files", "get", "--params", params])
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    parents = data.get("parents") or []
    return parents[0] if parents else None


def _format_size(size_bytes: str | int | None) -> str:
    if not size_bytes:
        return "?"
    try:
        n = int(size_bytes)
    except (TypeError, ValueError):
        return "?"
    if n >= 1 << 30:
        return f"{n / (1 << 30):.1f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.0f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.0f} kB"
    return f"{n} B"


def prompt_pick_video(videos: list[dict]) -> dict:
    """Print a numbered menu, read a 1-based choice from stdin, return the chosen file dict.

    Raises SystemExit if the user declines, the input is invalid, or stdin is closed
    (e.g. running non-interactively).
    """
    if not videos:
        raise SystemExit("No video files found in this folder.")

    print("", file=sys.stderr)
    print(f"Found {len(videos)} video(s) in folder:", file=sys.stderr)
    for i, v in enumerate(videos, 1):
        size = _format_size(v.get("size"))
        print(f"  [{i}] {v.get('name')}  ({size})", file=sys.stderr)
    print("", file=sys.stderr)
    print(f"Pick one [1-{len(videos)}] (or empty to cancel): ", end="", file=sys.stderr, flush=True)

    if not sys.stdin.isatty():
        raise SystemExit(
            "Drive folder URL requires an interactive choice, but stdin is not a TTY. "
            "Re-invoke /watch with a direct file URL "
            "(https://drive.google.com/file/d/<FILE_ID>/view) instead."
        )

    try:
        raw = sys.stdin.readline().strip()
    except KeyboardInterrupt:
        raise SystemExit("Cancelled.")

    if not raw:
        raise SystemExit("No selection — cancelled.")

    try:
        idx = int(raw)
    except ValueError:
        raise SystemExit(f"Invalid selection: {raw!r}")

    if not 1 <= idx <= len(videos):
        raise SystemExit(f"Selection out of range: {idx} (expected 1-{len(videos)})")

    return videos[idx - 1]
