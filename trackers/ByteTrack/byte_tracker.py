import torch

from trackers import BaseTracker
from core import BoundingBox, TrackObject
from .origin_tracker import BYTETracker


class ByteTracker(BaseTracker):
    def __init__(self, video_shape: tuple[int, int], fps: int):
        self.video_shape = video_shape
        self.fps = fps
        self._tracker = BYTETracker(
            track_thresh=0.5,
            track_buffer=50,
            mot20=False,
            match_thresh=0.7,
            frame_rate=fps
        )

    def update(self, bounding_boxes_batch: list[list[BoundingBox]]) -> list[list[TrackObject]]:
        res = []
        for bbs in bounding_boxes_batch:
            if bbs:
                pred = torch.vstack([
                    bb.to_tensor()
                    for bb in bbs
                ])
            else:
                pred = torch.empty((0, 5))
            online_targets = self._tracker.update(pred, self.video_shape)
            online_bbs = [
                t.to_track_object()
                for t in online_targets
            ]
            res.append(online_bbs)
        return res
