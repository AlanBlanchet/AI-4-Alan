import os
from pathlib import Path

import cv2
from pydantic import BaseModel, PrivateAttr


class VideoManager(BaseModel):
    log_path: Path
    shape: tuple

    _writer: cv2.VideoWriter = PrivateAttr(None)
    _video_p: Path = PrivateAttr(None)
    _active: bool = PrivateAttr(default=False)

    def create_file(self):
        if self._active:
            raise ValueError("Video file was not closed from previous recording")

        i = 1
        while (self.log_path / f"eval_{i}.mp4").exists():
            i += 1

        self._video_p = self.log_path / f"eval_{i}.avi"
        self._writer = cv2.VideoWriter(
            str(self._video_p), cv2.VideoWriter_fourcc(*"MPEG"), 30, self.shape
        )
        self._active = True

    def write(self, data):
        if self._writer is None:
            self.create_file()
        self._writer.write(data)

    def save(self):
        # Save the avi file
        self._writer.release()
        # Convert to mp4
        mp4 = self._video_p.with_suffix(".mp4")
        os.system(
            f"ffmpeg -hide_banner -i {str(self._video_p)} -c:v libx264 {mp4} 2> {str(self.log_path / 'ffmpeg.log')}"
        )
        os.remove(self._video_p)
        self._active = False
