"""
Benchmark: All SpinePose optimisations combined vs none.
"""

import time
import cv2
import mediapipe
import numpy as np
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

try:
    from spinepose import SpinePoseEstimator
except ImportError:
    print("spinepose not installed")
    sys.exit(1)

from Utility.video_utility import preprocess_video
from exercises.deadlift_analysis import EMA_ALPHA

VIDEO_PATH = "/Users/kostiuk_nazarii/Desktop/FYP_FitVision/model/inputs/raw/deadlift_me_test.mov"


def run_benchmark(optimised: bool):
    video_path = preprocess_video(VIDEO_PATH)
    spine_estimator = SpinePoseEstimator(device="cpu")

    mp_pose = mediapipe.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    INTERVAL = 2 if optimised else 1
    frame_idx = 0
    spine_kps_smoothed = None

    label = "WITH optimisations" if optimised else "WITHOUT optimisations"
    start = time.time()

    with tqdm(total=total_frames, desc=label, unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            pbar.update(1)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if frame_idx % INTERVAL == 0 and results.pose_landmarks:
                if optimised:
                    small = cv2.resize(frame, (width // 2, height // 2))
                    lms = results.pose_landmarks.landmark
                    xs = [lm.x * width for lm in lms]
                    ys = [lm.y * height for lm in lms]
                    x1 = max(0, int(min(xs)) - 150)
                    y1 = max(0, int(min(ys)) - 150)
                    x2 = min(width, int(max(xs)) + 150)
                    y2 = min(height, int(max(ys)) + 150)
                    bbox = np.array([[x1//2, y1//2, x2//2, y2//2]], dtype=np.float32)
                    kps, scs = spine_estimator(small, bboxes=bbox)
                else:
                    kps, scs = spine_estimator(frame)

                if len(kps) > 0:
                    raw_kps = kps[0].copy()
                    if optimised:
                        raw_kps[:, 0] *= 2
                        raw_kps[:, 1] *= 2
                    if spine_kps_smoothed is None:
                        spine_kps_smoothed = raw_kps.copy()
                    else:
                        spine_kps_smoothed = EMA_ALPHA * raw_kps + (1 - EMA_ALPHA) * spine_kps_smoothed

    elapsed = time.time() - start
    cap.release()
    pose.close()
    return elapsed, total_frames


print("=" * 50)
print("Running WITHOUT any optimisations...")
t1, frames = run_benchmark(optimised=False)
print(f"  Total time : {t1:.1f}s  |  {frames/t1:.1f} fps")

print()
print("Running WITH all optimisations...")
t2, frames = run_benchmark(optimised=True)
print(f"  Total time : {t2:.1f}s  |  {frames/t2:.1f} fps")

reduction = (t1 - t2) / t1 * 100
print()
print(f"Result: {reduction:.1f}% faster with optimisations")
print(f"        {t1:.1f}s -> {t2:.1f}s")
print("=" * 50)
