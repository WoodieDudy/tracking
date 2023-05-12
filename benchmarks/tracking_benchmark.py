import os
from typing import Optional, Callable, Sequence
from pathlib import Path
import time

from tqdm import tqdm
import pandas

from benchmarks.base import ITrackWriter, IBenchmark, IMOTAMetrics
from mot.mot_loading import load_mot_datas, MOTData
from trackers.base_tracker import BaseTracker
from utils import images_reader_by_batch


class TrackingBenchmark(IBenchmark):
    def __init__(self, detector, tracker_factory: Callable[[Path, Sequence], BaseTracker],
                 dataset_path: Path,
                 save_path: Path,
                 batch_size: int = 32,
                 video_writer_factory: Optional[Callable[[Path, Sequence], ITrackWriter]] = None,
                 mota_metrics: Optional[IMOTAMetrics] = None,
                 description: Optional[str] = None):
        self.detector = detector
        self.tracker_factory = tracker_factory
        self.dataset_path = dataset_path
        self.video_writer_factory = video_writer_factory
        self.mota_metrics = mota_metrics
        self.batch_size = batch_size
        self.description = description
        self.save_path = save_path

        self._video_writer = None

    def _make_video_writer(self, mot_data: MOTData):
        if self.video_writer_factory is not None:
            self._video_writer = self.video_writer_factory(mot_data.root_dir, mot_data.sequence)

    def _write_video(self, frame, tracking_objects):
        if self._video_writer is not None:
            self._video_writer.write(frame, tracking_objects)

    def _release_video_writer(self):
        if self._video_writer is not None:
            self._video_writer.release()

    def run(self):
        if self.description:
            print(self.description)
        total = len([d for d in self.dataset_path.glob('*') if d.is_dir()])
        timings = []
        boxes = []
        for mot_data in tqdm(load_mot_datas(self.dataset_path), desc='Videos', total=total):
            mot_data: MOTData

            tracker = self.tracker_factory(mot_data.root_dir, mot_data.sequence)
            gt_data = mot_data.gt_data

            self._make_video_writer(mot_data)
            batches_count = len(os.listdir(mot_data.image_dir)) // self.batch_size
            for batch_i, frames_batch in tqdm(enumerate(images_reader_by_batch(mot_data.image_dir, self.batch_size)), total=batches_count):
                batch_preds = self.detector.inference(frames_batch)
                boxes.extend([len(b) for b in batch_preds])
                start_time = time.time()
                tracking_objects_batch = tracker.update(frames_batch, batch_preds)
                timings.append(time.time() - start_time)

                for i, (frame, tracking_objects, bbs) in enumerate(zip(frames_batch, tracking_objects_batch, batch_preds)):
                    i = batch_i * self.batch_size + i
                    gt_frame = gt_data[gt_data['frame'] == i]

                    if self.mota_metrics:
                        self.mota_metrics.update(gt_frame, tracking_objects)

                    self._write_video(frame, tracking_objects)
            self._release_video_writer()

        if self.mota_metrics:
            metrics = self.mota_metrics.get_metrics()
            print(metrics)
            # metrics.columns = ['dataset', 'num_frames', 'mota', 'motp', 'idf1', 'precision', 'recall']
            metrics['fps'] = 1 / (sum(timings) / len(timings) / self.batch_size)
            metrics['avg boxes'] = sum(boxes) / len(boxes)
            metrics.to_csv(self.save_path, index=False)
