import dataclasses
import os
from typing import Iterable

import pandas as pd

from utils import parse_config_file, Sequence


@dataclasses.dataclass(frozen=True)
class MOTData:
    root_dir: str
    gt_data: pd.DataFrame
    image_dir: str
    sequence: Sequence

    @staticmethod
    def from_path(path: str) -> 'MOTData':
        gt_path = os.path.join(path, 'gt/gt.txt')

        gt_data = pd.read_csv(gt_path,
                              names=['frame', 'id', 'x', 'y', 'w', 'h', 'confidence', 'class', 'visibility'],
                              delimiter=',', header=None)
        config = parse_config_file(os.path.join(path, 'seqinfo.ini'))
        image_dir = os.path.join(path, config.imDir)
        return MOTData(path, gt_data, image_dir, config)


def load_mot_datas(path: str) -> Iterable[MOTData]:
    for exp_name in os.listdir(path):
        exp_path = os.path.join(path, exp_name)
        if os.path.isdir(exp_path):
            yield MOTData.from_path(exp_path)
