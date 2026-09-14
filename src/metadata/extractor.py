"""Extract audio metadata from FFprobe and ExifTool."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _run_json_command(command: list[str]) -> dict[str, Any] | list[Any]:
    """Run a metadata command and decode its JSON output."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        return {"error": f"Metadata tool not found: {command[0]}", "error_type": type(exc).__name__}
    except subprocess.CalledProcessError as exc:
        message = (exc.stderr or exc.stdout or str(exc)).strip()
        return {"error": f"{command[0]} failed: {message}", "error_type": type(exc).__name__}

    try:
        decoded = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {"error": f"{command[0]} returned invalid JSON: {exc}", "error_type": type(exc).__name__}
    return decoded


def extract_ffprobe_metadata(audio_path: Path) -> dict[str, Any]:
    """Extract container and stream metadata with FFprobe."""
    result = _run_json_command(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(Path(audio_path)),
        ]
    )
    return result if isinstance(result, dict) else {"data": result}


def extract_exiftool_metadata(audio_path: Path) -> dict[str, Any]:
    """Extract file and embedded metadata with ExifTool."""
    result = _run_json_command(["exiftool", "-j", str(Path(audio_path))])
    if isinstance(result, list):
        return result[0] if result and isinstance(result[0], dict) else {"data": result}
    return result


def _first_value(*values: Any) -> Any:
    """Return the first non-empty metadata value."""
    return next((value for value in values if value not in (None, "")), None)


def extract_metadata(audio_path: Path) -> dict[str, Any]:
    """Combine FFprobe and ExifTool metadata into a normalized forensic record."""
    path = Path(audio_path)
    ffprobe = extract_ffprobe_metadata(path)
    exiftool = extract_exiftool_metadata(path)
    format_info = ffprobe.get("format", {})
    streams = ffprobe.get("streams", [])
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        streams[0] if streams else {},
    )

    return {
        "file_name": path.name,
        "file_size": _first_value(format_info.get("size"), exiftool.get("FileSize")),
        "format": _first_value(format_info.get("format_name"), exiftool.get("FileTypeExtension")),
        "duration": _first_value(format_info.get("duration"), exiftool.get("Duration")),
        "codec": _first_value(audio_stream.get("codec_name"), exiftool.get("AudioCodec")),
        "sample_rate": _first_value(audio_stream.get("sample_rate"), exiftool.get("SampleRate")),
        "channels": _first_value(audio_stream.get("channels"), exiftool.get("Channels")),
        "bitrate": _first_value(
            format_info.get("bit_rate"),
            audio_stream.get("bit_rate"),
            exiftool.get("AudioBitrate"),
        ),
        "encoder": _first_value(
            format_info.get("tags", {}).get("encoder"),
            audio_stream.get("tags", {}).get("encoder"),
            exiftool.get("Encoder"),
        ),
        "creation_time": _first_value(
            format_info.get("tags", {}).get("creation_time"),
            exiftool.get("CreateDate"),
            exiftool.get("FileCreateDate"),
        ),
        "modification_time": _first_value(
            exiftool.get("ModifyDate"),
            exiftool.get("FileModifyDate"),
        ),
        "device_info": _first_value(
            exiftool.get("Make"),
            exiftool.get("Model"),
            exiftool.get("DeviceManufacturer"),
            exiftool.get("DeviceModel"),
        ),
        "software": _first_value(
            exiftool.get("Software"),
            format_info.get("tags", {}).get("encoder"),
        ),
        "comments": _first_value(
            exiftool.get("Comment"),
            exiftool.get("Description"),
            format_info.get("tags", {}).get("comment"),
        ),
        "raw_ffprobe": ffprobe,
        "raw_exiftool": exiftool,
    }


def get_metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the key normalized fields used for quick forensic review."""
    fields = (
        "file_name",
        "file_size",
        "format",
        "duration",
        "codec",
        "sample_rate",
        "channels",
        "bitrate",
        "encoder",
        "creation_time",
        "modification_time",
        "device_info",
        "software",
        "comments",
    )
    return {field: metadata.get(field) for field in fields}

