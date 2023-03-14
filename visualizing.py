import os.path
from typing import ClassVar

import numpy as np
import cv2


class TrackVisualizer:
    def __init__(self):
        self._border_size = 6
        self._border_color = (100, 178, 25)
        self._text_color = (45, 200, 45)

    def draw_box(self, img, coords, border_color=None):
        border_color = border_color or self._border_color

        x1, y1, x2, y2 = coords
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, self._border_size)

    def draw_label(self, img, coords, label, down=True):
        x1, y1, x2, y2 = coords
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 1.0
        FONT_THICKNESS = 2
        bg_color = (255, 255, 255)
        label_color = (0, 0, 0)

        (text_bg_w, text_bg_h), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)

        if down:
            text_x1, text_y1 = x2, y2 - text_bg_h
        else:
            text_x1, text_y1 = x2, y1

        cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1, self._border_size)

        cv2.putText(img, label, (text_x1, text_y1), FONT, FONT_SCALE, label_color, 1, 2)

    def plot_tracking(self, image_rgb, bbs):
        for i, bb in enumerate(bbs):
            self.draw_box(image_rgb, bb.xyxy, border_color=(255, 0, 255))
            self.draw_label(image_rgb, bb.xyxy, bb.label, down=False)

        return image_rgb
