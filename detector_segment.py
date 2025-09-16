# detector_segment.py
import cv2
import numpy as np
from blob import Blob
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects

def split_mask_watershed(mask, min_area=150):
    """Return list of instance masks (binary) after watershed splitting, or [] if not split."""
    if mask is None:
        return []
    m = (mask > 0).astype(np.uint8)
    if m.sum() < min_area:
        return []
    # remove small noise
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    local_max = (dist == ndi.maximum_filter(dist, size=7))
    markers, n = ndi.label(local_max)
    if n <= 1:
        return []
    markers = markers.astype(np.int32)
    img3 = cv2.cvtColor((m*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.watershed(img3, markers)
    instances = []
    for lab in range(1, markers.max()+1):
        inst = (markers == lab).astype(np.uint8)
        if inst.sum() >= min_area:
            instances.append(inst)
    return instances

class SegmentDetector:
    """
    Simple segmentation-based detector:
    - Grayscale + background subtract or adaptive threshold
    - Connected components -> blobs
    - Try to split large masks with watershed
    """
    def __init__(self, min_area=200, bg_subtractor=None):
        self.min_area = min_area
        # if user passes a bg subtractor, use it; otherwise use adaptive-threshold fallback
        self.bg = bg_subtractor

    def detect(self, frame, frame_id=0):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.bg is not None:
            fgmask = self.bg.apply(frame)
            _, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        else:
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 25, 7)
        # morphological cleanup
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

        # remove tiny objects using skimage helper (robust)
        try:
            th_clean = remove_small_objects(th.astype(bool), min_size=self.min_area)
            th = (th_clean.astype(np.uint8) * 255).astype(np.uint8)
        except Exception:
            # fallback
            pass

        # connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)
        blobs = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            mask_crop = (labels == i).astype(np.uint8)
            roi = gray[y:y+h, x:x+w].copy()
            mask_crop = mask_crop[y:y+h, x:x+w]
            # try watershed split if large
            if area > self.min_area * 3:
                insts = split_mask_watershed(mask_crop, min_area=self.min_area)
                if len(insts) > 0:
                    for inst in insts:
                        ys, xs = inst.nonzero()
                        if len(xs) == 0: continue
                        x0, x1 = xs.min(), xs.max()
                        y0, y1 = ys.min(), ys.max()
                        w2 = x1 - x0 + 1
                        h2 = y1 - y0 + 1
                        roi2 = roi[y0:y1+1, x0:x1+1].copy()
                        mask2 = inst[y0:y1+1, x0:x1+1].copy()
                        blobs.append(Blob((x + x0, y + y0, w2, h2), 1.0, frame_id, roi=roi2, mask=mask2))
                    continue
            blobs.append(Blob((x, y, w, h), 1.0, frame_id, roi=roi, mask=mask_crop))
        return blobs
