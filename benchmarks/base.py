import pandas as pd

from core import TrackObject


class IBenchmark:
    def run(self): ...


class ITrackWriter:
    def write(self, frame, tracking_objects: list[TrackObject]): ...

    def release(self): ...


class IMOTAMetrics:
    def update(self, gt_frame: pd.DataFrame, tracking_objects: list[TrackObject]): ...

    def get_metrics(self) -> dict[str, float]: ...

    def save_metrics(self, path: str): ...
