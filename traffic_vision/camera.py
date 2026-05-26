from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from typing import Any


def parse_camera_source(source: str | int) -> str | int:
    """Parse Streamlit/CLI camera source input into an OpenCV-friendly value."""

    if isinstance(source, int):
        return source
    value = str(source).strip()
    if value.isdigit():
        return int(value)
    return value


def is_stream_url(source: str) -> bool:
    scheme = urlparse(source).scheme.lower()
    return scheme in {"http", "https", "rtsp", "rtmp"}


@dataclass(frozen=True)
class SourceValidation:
    ok: bool
    source: str | int
    message: str


def validate_camera_source(source: str | int) -> SourceValidation:
    """Check that OpenCV can open a webcam, video file, or IP stream source."""

    parsed_source = parse_camera_source(source)
    if isinstance(parsed_source, str):
        if not parsed_source:
            return SourceValidation(False, parsed_source, "Camera source is empty.")
        if not is_stream_url(parsed_source) and not Path(parsed_source).expanduser().exists():
            return SourceValidation(
                False,
                parsed_source,
                f"Video file does not exist: {Path(parsed_source).expanduser()}",
            )

    import cv2

    capture = cv2.VideoCapture(parsed_source)
    try:
        if not capture.isOpened():
            if isinstance(parsed_source, int):
                return SourceValidation(
                    False,
                    parsed_source,
                    f"OpenCV could not open local webcam source {parsed_source}. Check camera permissions and that no other app is using it.",
                )
            if isinstance(parsed_source, str) and is_stream_url(parsed_source):
                return SourceValidation(
                    False,
                    parsed_source,
                    f"OpenCV could not open stream URL {parsed_source!r}. Check the URL, network access, credentials, and whether the stream is HTTP/RTSP compatible.",
                )
            return SourceValidation(
                False,
                parsed_source,
                f"OpenCV could not open video file {parsed_source!r}. Check the path and codec support.",
            )
    finally:
        capture.release()

    if isinstance(parsed_source, int):
        message = f"OpenCV can open local webcam source {parsed_source}."
    elif isinstance(parsed_source, str) and is_stream_url(parsed_source):
        message = f"OpenCV can open stream URL {parsed_source!r}."
    else:
        message = f"OpenCV can open video file {parsed_source!r}."
    return SourceValidation(True, parsed_source, message)


@dataclass
class CameraConfig:
    source: str | int = 0
    width: int | None = None
    height: int | None = None


class CameraStream:
    """Thin wrapper around OpenCV VideoCapture for cameras, files, and RTSP URLs."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: Any | None = None

    def open(self) -> None:
        import cv2

        validation = validate_camera_source(self.config.source)
        if not validation.ok:
            raise RuntimeError(validation.message)
        source = validation.source
        self._capture = cv2.VideoCapture(source)
        if self.config.width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        if self.config.height:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if not self._capture.isOpened():
            hint = f" ({Path(source).expanduser()})" if isinstance(source, str) else ""
            raise RuntimeError(f"Unable to open camera source {source!r}{hint}")

    def frames(self) -> Iterator[Any]:
        if self._capture is None:
            self.open()
        assert self._capture is not None
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            yield frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "CameraStream":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
