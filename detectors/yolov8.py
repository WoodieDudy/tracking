from ultralytics import YOLO

from detectors.base_detector import BaseDetector
from core import BoundingBox


class Yolov8Detector(BaseDetector):
    def __init__(self):
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.fp16 = False
        self.device = 'cpu'
        self.test_size = (640, 640)

        self._model = YOLO('/Users/georgy/Trash/yolov8m.pt')

        # self._model.classes = [0]
        # self._model.confidence = 0.2

    def inference(self, images_list):
        preds_of_frames = self._model(images_list)
        res = []
        for bbs_of_frames in preds_of_frames:
            bbs = []
            for xyxy, cls, conf in zip(bbs_of_frames.boxes.xyxy, bbs_of_frames.boxes.cls, bbs_of_frames.boxes.conf):
                bbs.append(BoundingBox(*xyxy, conf))
            res.append(bbs)

        return res
