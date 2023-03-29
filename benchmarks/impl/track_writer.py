from __future__ import annotations

from os import PathLike

import cv2

from benchmarks.base import ITrackWriter
from utils import Sequence
from visualizing import TrackVisualizer


class TrackWriter(ITrackWriter):
    def __init__(self, config: Sequence, output_path: str | PathLike = "result.mp4",
                 track_visualizer: TrackVisualizer = TrackVisualizer()):
        self.track_visualizer = track_visualizer
        self.video_writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            config.frameRate,
            (config.imWidth, config.imHeight)
        )

    def write(self, frame, tracking_objects):
        self.track_visualizer.plot_tracking(frame, tracking_objects)
        self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def release(self):
        self.video_writer.release()
        self.video_writer = None
