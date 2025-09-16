# blob.py
import numpy as np

class Blob:
    def __init__(self, bbox, score, frame_id, roi=None, mask=None):
        # bbox: (x,y,w,h)
        self.x, self.y, self.w, self.h = map(int, bbox)
        self.score = float(score)
        self.frame_id = int(frame_id)
        self.roi = roi  # grayscale crop
        self.mask = mask  # binary mask crop (same shape as roi)

    def centroid(self):
        return (self.x + self.w/2.0, self.y + self.h/2.0)

    def as_bbox(self):
        return (self.x, self.y, self.w, self.h)
