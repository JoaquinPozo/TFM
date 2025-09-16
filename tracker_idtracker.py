# tracker_idtracker.py
from detector_segment import SegmentDetector

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

# -------------------------------
# Kalman Filter for motion ()
# -------------------------------
class KalmanBoxTracker:
    def __init__(self, bbox, track_id, descriptor):
        self.track_id = track_id
        self.descriptor = descriptor
        self.skipped_frames = 0

        # (x, y, w, h) format
        x, y, w, h = bbox
        self.kf = cv2.KalmanFilter(8, 4)  # state: [x,y,w,h, vx,vy,vw,vh], meas: [x,y,w,h]
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i+4] = 1.0
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1.0
        self.kf.measurementMatrix[1, 1] = 1.0
        self.kf.measurementMatrix[2, 2] = 1.0
        self.kf.measurementMatrix[3, 3] = 1.0

        self.kf.statePre[:4, 0] = np.array([x, y, w, h], dtype=np.float32)
        self.kf.statePost[:4, 0] = np.array([x, y, w, h], dtype=np.float32)

        self.trace = deque(maxlen=50)  # trajectory history

    def predict(self):
        pred = self.kf.predict()
        return pred[:4].flatten()

    def update(self, bbox, descriptor):
        meas = np.array(bbox, dtype=np.float32).reshape(-1, 1)
        self.kf.correct(meas)
        self.descriptor = descriptor
        self.skipped_frames = 0
        self.trace.append(bbox)


# -------------------------------
# Tracker Manager
# -------------------------------
class IDTracker:
    def __init__(self, max_age=30, min_hits=3, descriptor_thresh=0.5):
        self.max_age = max_age
        self.min_hits = min_hits
        self.descriptor_thresh = descriptor_thresh
        self.next_id = 0
        self.tracks = []

    def _compute_descriptor(self, roi):
        """Simple fingerprint: normalized grayscale histogram"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _descriptor_distance(self, d1, d2):
        return cv2.compareHist(d1.astype("float32"), d2.astype("float32"), cv2.HISTCMP_BHATTACHARYYA)

    def update(self, detections, frame):
        """
        detections: list of [x, y, w, h] bounding boxes
        frame: current video frame
        """
        if len(self.tracks) == 0:
            # Initialize new tracks
            for det in detections:
                x, y, w, h = det
                roi = frame[int(y):int(y+h), int(x):int(x+w)]
                if roi.size == 0:
                    continue
                desc = self._compute_descriptor(roi)
                self.tracks.append(KalmanBoxTracker(det, self.next_id, desc))
                self.next_id += 1
            return []

        # Predict positions of existing tracks
        predictions = [trk.predict() for trk in self.tracks]

        # Build cost matrix using motion + descriptor
        cost_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        for i, trk in enumerate(self.tracks):
            for j, det in enumerate(detections):
                x, y, w, h = det
                roi = frame[int(y):int(y+h), int(x):int(x+w)]
                if roi.size == 0:
                    desc = trk.descriptor
                else:
                    desc = self._compute_descriptor(roi)

                # Motion cost (IoU distance)
                box_pred = predictions[i]
                iou_dist = 1 - self._iou(box_pred, det)

                # Appearance cost
                desc_dist = self._descriptor_distance(trk.descriptor, desc)

                cost_matrix[i, j] = 0.5 * iou_dist + 0.5 * desc_dist

        # Hungarian assignment
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        assigned_tracks = set()
        assigned_detections = set()

        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] > self.descriptor_thresh:
                continue
            trk = self.tracks[r]
            det = detections[c]
            x, y, w, h = det
            roi = frame[int(y):int(y+h), int(x):int(x+w)]
            if roi.size == 0:
                continue
            desc = self._compute_descriptor(roi)
            trk.update(det, desc)
            assigned_tracks.add(r)
            assigned_detections.add(c)

        # Create new tracks for unassigned detections
        for j, det in enumerate(detections):
            if j not in assigned_detections:
                x, y, w, h = det
                roi = frame[int(y):int(y+h), int(x):int(x+w)]
                if roi.size == 0:
                    continue
                desc = self._compute_descriptor(roi)
                self.tracks.append(KalmanBoxTracker(det, self.next_id, desc))
                self.next_id += 1

        # Remove stale tracks
        alive_tracks = []
        for i, trk in enumerate(self.tracks):
            if i not in assigned_tracks:
                trk.skipped_frames += 1
            if trk.skipped_frames <= self.max_age:
                alive_tracks.append(trk)
        self.tracks = alive_tracks

        # Return current active tracks
        results = []
        for trk in self.tracks:
            x, y, w, h = trk.kf.statePost[:4].flatten()
            results.append([int(x), int(y), int(w), int(h), trk.track_id])
        return results

    @staticmethod
    def _iou(bb1, bb2):
        # bb: [x,y,w,h]
        x1, y1, w1, h1 = bb1
        x2, y2, w2, h2 = bb2
        xx1 = max(x1, x2)
        yy1 = max(y1, y2)
        xx2 = min(x1+w1, x2+w2)
        yy2 = min(y1+h1, y2+h2)
        inter = max(0, xx2-xx1) * max(0, yy2-yy1)
        union = w1*h1 + w2*h2 - inter
        return inter / union if union > 0 else 0
