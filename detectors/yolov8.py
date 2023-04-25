from ultralytics import YOLO

from detectors.base_detector import BaseDetector
from core import BoundingBox


class Yolov8Detector(BaseDetector):
    def __init__(self):
        self._model = YOLO('assets/weights/yolov8m.pt')

    def inference(self, images_list):
        preds_of_frames = self._model.predict(source=images_list, classes = [0], conf= 0.1, verbose=False)
        res = []
        for bbs_of_frames in preds_of_frames:
            bbs = []
            for xyxy, cls, conf in zip(bbs_of_frames.boxes.xyxy, bbs_of_frames.boxes.cls, bbs_of_frames.boxes.conf):
                bbs.append(BoundingBox(*xyxy, conf.cpu(), cls.cpu()))
            res.append(bbs)

        return res
