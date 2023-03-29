import motmetrics as mm
import numpy as np
import pandas as pd

from benchmarks.base import IMOTAMetrics
from core import TrackObject


class MOTAMetrics(IMOTAMetrics):
    def __init__(self, max_iou: float = 0.5):
        self.acc = mm.MOTAccumulator(auto_id=True)
        self.max_iou = max_iou

    def update(self, gt_frame: pd.DataFrame, tracking_objects: list[TrackObject]):
        gt_bbs = gt_frame[['x', 'y', 'w', 'h']].values
        tr_bbs = np.array([list(obj.xywh) for obj in tracking_objects])
        distances = mm.distances.iou_matrix(gt_bbs, tr_bbs, max_iou=self.max_iou)

        self.acc.update(gt_frame['id'].values, [obj.track_id for obj in tracking_objects], distances)

    def get_metrics(self) -> pd.DataFrame:
        mh = mm.metrics.create()
        return mh.compute(self.acc, metrics=['num_frames', 'mota', 'motp', 'idf1', 'precision', 'recall'], name='mot20')

    def save_metrics(self, path: str):
        summary = self.get_metrics()
        summary.to_csv(path)
