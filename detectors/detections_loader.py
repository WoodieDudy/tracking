from pathlib import Path

from detectors.base_detector import BaseDetector
from core import BoundingBox
from mot.mot_loading import MOTData, load_mot_datas

import pandas as pd
from tqdm import tqdm


class DetectionsLoader(BaseDetector):
    def __init__(self, dataset_path: Path, file_prefix: str = 'gt'):
        self.dataset_path = dataset_path
        self.file_prefix = file_prefix
        self.frame_id = 0
        self.iterator = self._iterator()

    def _iterator(self):
        for mot_data in load_mot_datas(self.dataset_path, file_prefix=self.file_prefix):
            mot_data: MOTData
            gt_data = mot_data.gt_data
            video_len = mot_data.sequence.seqLength

            for frame_id in tqdm(range(1, video_len + 1)):
                frame_gt: pd.DataFrame = gt_data[gt_data['frame'] == frame_id]
                bbs = []
                for x, y, w, h in zip(frame_gt['x'], frame_gt['y'], frame_gt['w'], frame_gt['h']):
                    bbs.append(BoundingBox(x, y, x + w, y + h, 1))
                yield bbs
        
    def inference(self, images_list):
        return [
            next(self.iterator)
            for _ in range(len(images_list))
        ]
