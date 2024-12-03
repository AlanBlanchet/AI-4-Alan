import os
import subprocess
from io import FileIO
from pathlib import Path

import cv2
from pydantic import PrivateAttr

from ...configs.base import BaseMP


class VideoManager(BaseMP):
    log_name = "Video"

    log_path: Path
    shape: tuple
    stream: bool = False
    open_stream: bool = False
    """Open the stream in a new window using ffplay"""

    _writer: FileIO = PrivateAttr(None)
    _video_p: Path = PrivateAttr(None)

    @BaseMP.watch
    def create_file(self):
        def mk_name(i, ext: str = ""):
            if self.stream:
                return f"{i}_stream"
            else:
                return f"{i}{ext}"

        i = 1
        while (self.log_path / mk_name(i, ext=".mp4")).exists():
            i += 1

        self.log_path.mkdir(parents=True, exist_ok=True)
        if self.stream:
            if self._writer is not None:
                # Streaming keeps the previous fifo
                return

            self._video_p = self.log_path / f"{self.worker_id}_stream"
            path = str(self._video_p)
            os.mkfifo(path)
            fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
            self._writer = open(fd, "wb", buffering=0)
            self.info(f"Creating fifo for streaming {path}")

            if self.open_stream:
                subprocess.Popen(
                    f"ffplay -an -f mjpeg -i {path} -loglevel error -sws_flags neighbor",
                    shell=True,
                    env={**os.environ, "SDL_RENDER_SCALE_QUALITY": "0"},
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        else:
            if self.open_stream:
                self.warn_once(
                    f"Cannot open the stream in a window when {self.stream=}"
                )

            self._video_p = self.log_path / mk_name(i, ext=".avi")
            path = str(self._video_p)
            self._writer = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"MPEG"), 30, self.shape
            )

    @BaseMP.watch
    def write(self, data):
        if self._writer is None:
            self.create_file()

        if self.stream:
            _, encoded_frame = cv2.imencode(".jpg", data)
            self._writer.write(b"--frame\r\n")
            self._writer.write(b"Content-Type: image/jpeg\r\n\r\n")
            self._writer.write(encoded_frame.tobytes())
            self._writer.write(b"\r\n")
        else:
            self._writer.write(data)

    def save(self):
        # Save the avi file
        self._writer.release()
        # Convert to mp4
        mp4 = self._video_p.with_suffix(".mp4")
        subprocess.run(
            f"ffmpeg -hide_banner -i {str(self._video_p)} -c:v libx264 {mp4} 2> {str(self.log_path / 'ffmpeg.log')}",
            shell=True,
        )
        os.remove(self._video_p)
        self._active = False
