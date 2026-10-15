from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2


class VideoSource(Protocol):
    def read(self):
        ...

    def release(self) -> None:
        ...


class OpenCVVideoSource:
    def __init__(self, source: str | Path | int) -> None:
        self.source = source
        self.capture = cv2.VideoCapture(str(source) if isinstance(source, Path) else source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Impossibile aprire la sorgente video: {source}")

    @property
    def width(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)

    def read(self):
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()


def open_video_file(path: str | Path) -> OpenCVVideoSource:
    return OpenCVVideoSource(Path(path))


def open_usb_camera(index: int = 0) -> OpenCVVideoSource:
    return OpenCVVideoSource(index)


def open_stream(url: str) -> OpenCVVideoSource:
    return OpenCVVideoSource(url)
