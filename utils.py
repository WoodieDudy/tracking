import configparser
import os
from collections import namedtuple

import cv2

Sequence = namedtuple('Sequence', ['name', 'imDir', 'frameRate', 'seqLength', 'imWidth', 'imHeight', 'imExt'])


def parse_config_file(config_file_path) -> Sequence:
    config = configparser.ConfigParser()

    config.read(config_file_path)

    name = config.get('Sequence', 'name')
    imDir = config.get('Sequence', 'imDir')
    frameRate = config.getint('Sequence', 'frameRate')
    seqLength = config.getint('Sequence', 'seqLength')
    imWidth = config.getint('Sequence', 'imWidth')
    imHeight = config.getint('Sequence', 'imHeight')
    imExt = config.get('Sequence', 'imExt')

    return Sequence(name, imDir, frameRate, seqLength, imWidth, imHeight, imExt)


def images_reader_by_batch(dir, batch_size):
    files = os.listdir(dir)
    files.sort(key=lambda x: int(x.split('.')[0]))
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        images = [
            cv2.cvtColor(cv2.imread(os.path.join(dir, file)), cv2.COLOR_BGR2RGB)
            for file in batch
        ]
        yield images
