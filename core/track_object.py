import dataclasses

from . import BoundingBox


@dataclasses.dataclass
class TrackObject:
    bounding_box: BoundingBox
    track_id: int
    cls_name: str = None

    @property
    def xyxy(self):
        return self.bounding_box.xyxy

    @property
    def label(self):
        return f"tid: {self.track_id}, {self.bounding_box.cls_id}"
