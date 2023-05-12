from pathlib import Path
import os

from benchmarks.becnhmark_runner import BenchmarkRunner
from benchmarks.impl.mota_metrics import MOTAMetrics
from benchmarks.impl.track_writer import TrackWriter
from benchmarks.tracking_benchmark import TrackingBenchmark
from detectors import Yolov8Detector, DetectionsLoader
from trackers import ByteTracker, DeepSort
from trackers.deep_sort.tracker import RealDeepSort
from trackers import DeepSortReid

sportsmot_val = Path('/home/gk/projects/nir_tracking/datasets/sportsmot_publish/dataset/val')
mot20_train = Path('/home/gk/projects/nir_tracking/datasets/MOT20/train')
mot17_train = Path('/home/gk/projects/nir_tracking/datasets/MOT17/train')
batch_size = 64

videos_path = Path('videos')
save_metrics_path = Path('metrics')
os.makedirs(str(videos_path), exist_ok=True)
os.makedirs(str(save_metrics_path), exist_ok=True)


bytetrack_benchmark1 = TrackingBenchmark(
    save_path=save_metrics_path / "ByteTrack_sportsmot_val_gt.csv",
    detector=DetectionsLoader(Path(sportsmot_val), 'gt'),
    tracker_factory=lambda root_dir, sequence: ByteTracker(fps=sequence.frameRate,
                                                           video_shape=(sequence.imHeight, sequence.imWidth)),
    dataset_path=sportsmot_val,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'ByteTrack_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='ByteTrack'
)

bytetrack_benchmark2 = TrackingBenchmark(
    save_path=save_metrics_path / "ByteTrack_sportsmot_val_yolov8.csv",
    detector=Yolov8Detector(),
    tracker_factory=lambda root_dir, sequence: ByteTracker(fps=sequence.frameRate,
                                                           video_shape=(sequence.imHeight, sequence.imWidth)),
    dataset_path=sportsmot_val,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'ByteTrack_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='ByteTrack'
)

bytetrack_benchmark3 = TrackingBenchmark(
    save_path=save_metrics_path / "ByteTrack_mot20_train_gt.csv",
    detector=DetectionsLoader(Path(mot20_train), 'gt'),
    tracker_factory=lambda root_dir, sequence: ByteTracker(fps=sequence.frameRate,
                                                           video_shape=(sequence.imHeight, sequence.imWidth)),
    dataset_path=mot20_train,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'ByteTrack_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='ByteTrack'
)

bytetrack_benchmark4 = TrackingBenchmark(
    save_path=save_metrics_path / "ByteTrack_mot20_train_det.csv",
    detector=DetectionsLoader(Path(mot20_train), 'det'),
    tracker_factory=lambda root_dir, sequence: ByteTracker(fps=sequence.frameRate,
                                                           video_shape=(sequence.imHeight, sequence.imWidth)),
    dataset_path=mot20_train,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'ByteTrack_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='ByteTrack'
)

bytetrack_benchmark5 = TrackingBenchmark(
    save_path=save_metrics_path / "ByteTrack_mot20_train_yolov8.csv",
    detector=Yolov8Detector(),
    tracker_factory=lambda root_dir, sequence: ByteTracker(fps=sequence.frameRate,
                                                           video_shape=(sequence.imHeight, sequence.imWidth)),
    dataset_path=mot20_train,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'ByteTrack_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='ByteTrack'
)


bytetrack_benchmark6 = TrackingBenchmark(
    save_path=save_metrics_path / "ByteTrack_mot17_train_gt.csv",
    detector=DetectionsLoader(Path(mot17_train), 'gt'),
    tracker_factory=lambda root_dir, sequence: ByteTracker(fps=sequence.frameRate,
                                                           video_shape=(sequence.imHeight, sequence.imWidth)),
    dataset_path=mot17_train,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'ByteTrack_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='ByteTrack'
)

bytetrack_benchmark7 = TrackingBenchmark(
    save_path=save_metrics_path / "ByteTrack_mot17_train_det.csv",
    detector=DetectionsLoader(Path(mot17_train), 'det'),
    tracker_factory=lambda root_dir, sequence: ByteTracker(fps=sequence.frameRate,
                                                           video_shape=(sequence.imHeight, sequence.imWidth)),
    dataset_path=mot17_train,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'ByteTrack_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='ByteTrack'
)

bytetrack_benchmark8 = TrackingBenchmark(
    save_path=save_metrics_path / "ByteTrack_mot17_train_yolov8.csv",
    detector=Yolov8Detector(),
    tracker_factory=lambda root_dir, sequence: ByteTracker(fps=sequence.frameRate,
                                                           video_shape=(sequence.imHeight, sequence.imWidth)),
    dataset_path=mot17_train,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'ByteTrack_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='ByteTrack'
)

# --------------------------------------------------------------------------------

deepsort_benchmark1 = TrackingBenchmark(
    save_path=save_metrics_path / "DeepSORT_sportsmot_val_gt.csv",
    detector=DetectionsLoader(Path(sportsmot_val), 'gt'),
    tracker_factory=lambda root_dir, sequence: DeepSortReid(),
    dataset_path=sportsmot_val,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'DeepSORT_sportsmot_val_gt_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='DeepSort'
)

deepsort_benchmark2 = TrackingBenchmark(
    save_path=save_metrics_path / "DeepSORT_sportsmot_val_yolov8.csv",
    detector=Yolov8Detector(),
    tracker_factory=lambda root_dir, sequence: DeepSortReid(),
    dataset_path=sportsmot_val,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'DeepSORT_sportsmot_val_yolov8_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='DeepSort'
)

deepsort_benchmark3 = TrackingBenchmark(
    save_path=save_metrics_path / "DeepSORT_mot20_train_gt.csv",
    detector=DetectionsLoader(Path(mot20_train), 'gt'),
    tracker_factory=lambda root_dir, sequence: DeepSortReid(),
    dataset_path=mot20_train,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'DeepSORT_mot20_train_gt_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='DeepSort'
)

deepsort_benchmark4 = TrackingBenchmark(
    save_path=save_metrics_path / "DeepSORT_mot20_train_det.csv",
    detector=DetectionsLoader(Path(mot20_train), 'det'),
    tracker_factory=lambda root_dir, sequence: DeepSortReid(),
    dataset_path=mot20_train,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'DeepSORT_mot20_train_det_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='DeepSort'
)

deepsort_benchmark5 = TrackingBenchmark(
    save_path=save_metrics_path / "DeepSORT_mot20_train_yolov8.csv",
    detector=Yolov8Detector(),
    tracker_factory=lambda root_dir, sequence: DeepSortReid(),
    dataset_path=mot20_train,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'DeepSORT_mot20_train_yolov8_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='DeepSort'
)


deepsort_benchmark6 = TrackingBenchmark(
    save_path=save_metrics_path / "DeepSORT_mot17_train_gt.csv",
    detector=DetectionsLoader(Path(mot17_train), 'gt'),
    tracker_factory=lambda root_dir, sequence: DeepSortReid(),
    dataset_path=mot17_train,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'DeepSORT_mot17_train_gt_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='DeepSort'
)

deepsort_benchmark7 = TrackingBenchmark(
    save_path=save_metrics_path / "DeepSORT_mot17_train_det.csv",
    detector=DetectionsLoader(Path(mot17_train), 'det'),
    tracker_factory=lambda root_dir, sequence: DeepSortReid(),
    dataset_path=mot17_train,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'DeepSORT_mot17_train_det_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='DeepSort'
)

deepsort_benchmark8 = TrackingBenchmark(
    save_path=save_metrics_path / "DeepSORT_mot17_train_yolov8.csv",
    detector=Yolov8Detector(),
    tracker_factory=lambda root_dir, sequence: DeepSortReid(),
    dataset_path=mot17_train,
    batch_size=batch_size,
    video_writer_factory=lambda root_dir, sequence: TrackWriter(sequence, videos_path / f'DeepSORT_mot17_train_yolov8_{str(root_dir)[-2:]}_result.mp4'),
    mota_metrics=MOTAMetrics(max_iou=0.5),
    description='DeepSort'
)

runner = BenchmarkRunner([
    bytetrack_benchmark1,
    deepsort_benchmark1,

    bytetrack_benchmark2,
    bytetrack_benchmark3,
    bytetrack_benchmark4,
    bytetrack_benchmark5,
    bytetrack_benchmark6,
    bytetrack_benchmark7,
    bytetrack_benchmark8,

    deepsort_benchmark2,
    deepsort_benchmark3,
    deepsort_benchmark4,
    deepsort_benchmark5,
    deepsort_benchmark6,
    deepsort_benchmark7,
    deepsort_benchmark8,
])

runner.run()
