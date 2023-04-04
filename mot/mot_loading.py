import dataclasses
import os
from typing import Iterable
from pathlib import Path
import warnings

import pandas as pd

from utils import parse_config_file, Sequence


warnings.simplefilter(action='ignore', category=pd.errors.ParserWarning)


@dataclasses.dataclass(frozen=True)
class MOTData:
    root_dir: Path
    gt_data: pd.DataFrame
    image_dir: str
    sequence: Sequence

    @staticmethod
    def from_path(path: Path, file_prefix: str = 'gt') -> 'MOTData':
        gt_path = os.path.join(path, f'{file_prefix}/{file_prefix}.txt')

        gt_data = pd.read_csv(gt_path,
                              names=['frame', 'id', 'x', 'y', 'w', 'h', 'confidence', 'class', 'visibility'],
                              delimiter=',', index_col=False, header=None)
        config = parse_config_file(os.path.join(path, 'seqinfo.ini'))
        image_dir = os.path.join(path, config.imDir)
        return MOTData(path, gt_data, image_dir, config)


def load_mot_datas(path: Path, file_prefix: str = 'gt') -> Iterable[MOTData]:
    for exp_name in sorted(os.listdir(path)):
        exp_path = path / exp_name
        if os.path.isdir(exp_path):
            yield MOTData.from_path(exp_path, file_prefix)
