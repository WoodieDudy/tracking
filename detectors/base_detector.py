import abc

from core import BoundingBox


class BaseDetector(abc.ABC):
    @abc.abstractmethod
    def inference(self, images_list: list) -> list[list[BoundingBox]]: ...
