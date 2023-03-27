import os
import configparser
from collections import namedtuple

import pandas as pd
import cv2
from tqdm import tqdm
import motmetrics as mm
import numpy as np

from detectors import Yolov8Detector
from trackers import ByteTracker
from cv2wrappers import VideoCapture
from visualizing import TrackVisualizer
from utils import parse_config_file, images_reader_by_batch


# dataset_path = '/home/gk/projects/nir_tracking/datasets/sportsmot_publish/dataset/val'
dataset_path = '/home/gk/projects/nir_tracking/datasets/MOT20/train/'
batch_size = 64

acc = mm.MOTAccumulator(auto_id=True)

for exp_name in tqdm(os.listdir(dataset_path)[::-1]):
    try:
        exp_path = os.path.join(dataset_path, exp_name)
        if not os.path.isdir(exp_path):
            continue
        gt_path = os.path.join(exp_path, 'gt/gt.txt')

        gt_data = pd.read_csv(gt_path, names=['frame', 'id', 'x', 'y', 'w', 'h', 'confidence', 'class', 'visibility'], delimiter=',', header=None)

        config = parse_config_file(os.path.join(exp_path, 'seqinfo.ini'))

        detector = Yolov8Detector()
        tracker = ByteTracker(fps=25, video_shape=(config.imHeight, config.imWidth))
        # video_writer = cv2.VideoWriter(
        #     'res.mp4',
        #     cv2.VideoWriter_fourcc(*'mp4v'),
        #     config.frameRate,
        #     (config.imHeight, config.imWidth)
        # )
        visualizer = TrackVisualizer()
        for batch_i, frames_batch in enumerate(images_reader_by_batch(os.path.join(exp_path, 'img1'), batch_size)):
            batch_preds = detector.inference(frames_batch)
            tracking_objects_batch = tracker.update(batch_preds)

            for i, (frame, tracking_objects, bbs) in enumerate(zip(frames_batch, tracking_objects_batch, batch_preds)):
                i = batch_i * batch_size + i
                gt_frame = gt_data[gt_data['frame'] == i]

                gt_bbs = gt_frame[['x', 'y', 'w', 'h']].values
                tr_bbs = np.array([list(obj.xywh) for obj in tracking_objects])
                distances = mm.distances.iou_matrix(gt_bbs, tr_bbs, max_iou=0.5)

                acc.update(gt_frame['id'].values, [obj.track_id for obj in tracking_objects], distances)
                # visualizer.plot_tracking(frame, tracking_objects)
                # video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(e)


mh = mm.metrics.create()
summary = mh.compute(acc, metrics=['num_frames', 'mota', 'motp', 'idf1', 'precision', 'recall'], name='mot20')
print(summary)
summary_path = os.path.join(dataset_path, 'summary.csv')
summary.to_csv(summary_path)
