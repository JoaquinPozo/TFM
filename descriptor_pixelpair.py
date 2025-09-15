# descriptor_pixelpair.py
import numpy as np
import cv2

def compute_pixelpair_descriptor(
    roi, 
    distances=(1, 2, 3, 5, 7),
    n_directions=8,
    n_bins=16,
    normalize=True
):
    """
    Compute idTracker-inspired descriptor:
    - Pixel pair intensity sum (i1+i2)
    - Pixel pair intensity difference |i1-i2|
    - Gradient orientation histogram (extra fingerprint robustness)
    """

    if roi is None or roi.size == 0:
        return np.zeros(len(distances) * 2 * n_bins + n_bins, dtype=np.float32)

    # Ensure grayscale
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    img = roi.astype(np.int32)
    H, W = img.shape
    angles = [2 * np.pi * k / n_directions for k in range(n_directions)]
    offsets = []
    for d in distances:
        for ang in angles:
            dx = int(round(d * np.cos(ang)))
            dy = int(round(d * np.sin(ang)))
            if dx == 0 and dy == 0:
                dx = d
            offsets.append((dx, dy, d))

    # Hist binning ranges
    sum_bins = np.linspace(0, 510, n_bins + 1)   # since pixel values up to 255+255
    diff_bins = np.linspace(0, 255, n_bins + 1)

    sum_hist = np.zeros((len(distances), n_bins), dtype=np.float32)
    diff_hist = np.zeros((len(distances), n_bins), dtype=np.float32)

    pad = max(distances) + 1
    padded = np.pad(img, pad, mode="reflect")

    # --- Pixel-pair statistics ---
    for dx, dy, dval in offsets:
        x0 = pad + max(0, -dx)
        x1 = pad + max(0, -dx) + W
        y0 = pad + max(0, -dy)
        y1 = pad + max(0, -dy) + H

        x0s = pad + max(0, dx)
        x1s = pad + max(0, dx) + W
        y0s = pad + max(0, dy)
        y1s = pad + max(0, dy) + H

        A = padded[y0:y1, x0:x1].ravel()
        B = padded[y0s:y1s, x0s:x1s].ravel()
        S = A + B
        D = np.abs(A - B)

        d_index = distances.index(dval)
        s_counts, _ = np.histogram(S, bins=sum_bins)
        d_counts, _ = np.histogram(D, bins=diff_bins)
        sum_hist[d_index] += s_counts
        diff_hist[d_index] += d_counts

    # --- Gradient orientation histogram (like SIFT) ---
    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)

    orientation_hist, _ = np.histogram(
        ang.ravel(),
        bins=n_bins,
        range=(0, 2 * np.pi),
        weights=mag.ravel()
    )

    # --- Assemble descriptor ---
    vecs = []
    for i in range(len(distances)):
        s = sum_hist[i].astype(np.float32)
        d_ = diff_hist[i].astype(np.float32)
        if normalize:
            s = s / (s.sum() + 1e-8)
            d_ = d_ / (d_.sum() + 1e-8)
        vecs.append(s)
        vecs.append(d_)
    orientation_hist = orientation_hist.astype(np.float32)
    if normalize:
        orientation_hist /= (orientation_hist.sum() + 1e-8)
    vecs.append(orientation_hist)

    descriptor = np.concatenate(vecs).astype(np.float32)
    descriptor /= (np.linalg.norm(descriptor) + 1e-8)
    return descriptor
