import os.path
from typing import ClassVar

import numpy as np
import cv2

from core import TrackObject


class TrackVisualizer:
    def __init__(self):
        self._border_size = 6
        self._border_color = (100, 178, 25)
        self._text_color = (45, 200, 45)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_thickness = 2

    def draw_box(self, img, coords, border_color=None):
        border_color = border_color or self._border_color

        x1, y1, x2, y2 = coords
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, self._border_size)

    def draw_label(self, img, coords, label):
        x1, y1, x2, y2 = coords
        bg_color = (0, 0, 0)
        label_color = (255, 255, 255)

        (text_bg_w, text_bg_h), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
        text_x1, text_y1 = x1 + 2, y2 - text_bg_h - self._border_size

        cv2.rectangle(img, (text_x1, text_y1), (text_x1 + text_bg_w, text_y1 + text_bg_h + baseline // 2), bg_color, -1)
        cv2.putText(img, label, (text_x1, text_y1 + text_bg_h), self.font, self.font_scale, label_color, self.font_thickness, 2)

    def plot_tracking(self, image_rgb, bbs, box_color=(255, 0, 255)):
        for i, bb in enumerate(bbs):
            self.draw_box(image_rgb, bb.xyxy, border_color=box_color)
            if isinstance(bb, TrackObject):
                self.draw_label(image_rgb, bb.xyxy, bb.label)

        return image_rgb
