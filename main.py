# main.py
import cv2
import numpy as np
import csv
import os
from tqdm import tqdm
from tracker_idtracker import IDTracker

# ---------------------------
# Utility: assign unique colors per ID
# ---------------------------
def get_color(idx):
    np.random.seed(idx)
    return tuple(int(x) for x in np.random.randint(0, 255, 3))


def main():
    input_video = "TFM-main/In1.avi"
    output_video = "output_tracked35.mp4"
    output_csv = "trajectories.csv"

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return 

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    tracker = IDTracker(max_age=30, min_hits=3, descriptor_thresh=0.7)

    # CSV for saving results
    csv_file = open(output_csv, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "id", "x", "y", "w", "h"])

    for frame_id in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        # ---------------------------
        # Preprocessing to detect animals
        # ---------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 500:  # skip noise
                continue
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            detections.append([x, y, w_box, h_box])

        # ---------------------------
        # Update tracker
        # ---------------------------
        tracks = tracker.update(detections, frame)

        # ---------------------------
        # Draw results
        # ---------------------------
        for tr in tracks:
            x, y, w_box, h_box, track_id = tr
            color = get_color(track_id)

            # Bounding box + ID
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            cv2.putText(frame, f"ID {track_id}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw trajectory
            for tracker_obj in tracker.tracks:
                if tracker_obj.track_id == track_id:
                    pts = list(tracker_obj.trace)
                    for i in range(1, len(pts)):
                        p1 = (int(pts[i-1][0] + pts[i-1][2] / 2),
                              int(pts[i-1][1] + pts[i-1][3] / 2))
                        p2 = (int(pts[i][0] + pts[i][2] / 2),
                              int(pts[i][1] + pts[i][3] / 2))
                        cv2.line(frame, p1, p2, color, 2)

            # Save to CSV
            csv_writer.writerow([frame_id, track_id, x, y, w_box, h_box])

        out.write(frame)

    cap.release()
    out.release()
    csv_file.close()
    print(f"✅ Done! Results saved to {output_video} and {output_csv}")


if __name__ == "__main__":
    main()
