from deep_sort_realtime.deepsort_tracker import DeepSort as DeepSortTracker

from core import BoundingBox, TrackObject
from trackers.base_tracker import BaseTracker


class DeepSort(BaseTracker):
    def __init__(self):
        self._tracker = DeepSortTracker()

    def _get_xyxy(self, track):
        ret = track.to_tlwh()
        ret[2:] += ret[:2]
        return ret

    def _to_track_object(self, track) -> TrackObject:
        return TrackObject(
            bounding_box=BoundingBox(*self._get_xyxy(track), conf=track.get_det_conf()),
            track_id=track.track_id,
        )

    def update(self, frames: list, bounding_boxes_batch: list[list[BoundingBox]]) -> list[list[TrackObject]]:
        track_objects_batch = []

        for frame, bbs in zip(frames, bounding_boxes_batch):
            if bbs:
                pred = [bb.to_deepsort() for bb in bbs]
            else:
                pred = []

            online_targets = self._tracker.update_tracks(pred, frame=frame)
            online_bbs = [self._to_track_object(t) for t in online_targets if
                          t.is_confirmed() and t.get_det_conf() is not None]
            track_objects_batch.append(online_bbs)

        return track_objects_batch
