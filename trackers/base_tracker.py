import abc

from core import BoundingBox, TrackObject


class BaseTracker(abc.ABC):
    @abc.abstractmethod
    def update(self, frames: list, bounding_boxes: list[list[BoundingBox]]) -> list[list[TrackObject]]: ...
