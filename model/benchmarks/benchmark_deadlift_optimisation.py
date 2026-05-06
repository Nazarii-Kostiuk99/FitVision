# Benchmark: deadlift SpinePose optimisation
# benchmark_deadlift_optimisation.py

import time
import cv2
import mediapipe
import numpy as np
import sys
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from spinepose import SpinePoseEstimator
except ImportError:
    print("spinepose not installed")
    sys.exit(1)

from Utility.video_utility import preprocess_video
from exercises.deadlift_analysis import EMA_ALPHA

VIDEO_PATH = "/Users/kostiuk_nazarii/Desktop/project_videos/in/deadlifts/benchmark/deadlift_me.mov"  # 17 sec
GRAPH_OUT = (
    "/Users/kostiuk_nazarii/Desktop/project_MAINN/Figures/testing/scaling.png"
)


TEST_CONFIGS = [
    {"name": "Baseline", "interval": 1, "half_res": False, "bbox": False},
    {"name": "+ Frame interval", "interval": 2, "half_res": False, "bbox": False},
    {"name": "+ Half-resolution", "interval": 2, "half_res": True, "bbox": False},
    {"name": "+ Bounding-box bypass", "interval": 2, "half_res": True, "bbox": True},
]

# video durations (scaling experiment)
SCALE_DURATIONS = [5, 10, 15, 19]


def _spine_pass(spine, frame, width, height, cfg, results):

    if cfg["half_res"]:
        proc = cv2.resize(frame, (width // 2, height // 2))
        scale = 0.5
    else:
        proc = frame
        scale = 1.0

    if cfg["bbox"] and results.pose_landmarks:
        lms = results.pose_landmarks.landmark
        xs = [lm.x * width * scale for lm in lms]
        ys = [lm.y * height * scale for lm in lms]
        x1 = max(0, int(min(xs)) - 100)
        y1 = max(0, int(min(ys)) - 100)
        x2 = min(int(width * scale), int(max(xs)) + 100)
        y2 = min(int(height * scale), int(max(ys)) + 100)
        kps, _ = spine(proc, bboxes=np.array([[x1, y1, x2, y2]], dtype=np.float32))
    else:
        kps, _ = spine(proc)

    if len(kps) == 0:
        return None
    raw = kps[0].copy()
    if cfg["half_res"]:
        raw[:, 0] *= 2
        raw[:, 1] *= 2
    return raw


# processes path with the given optimisation config + return elapsed_seconds frames_processed peak_ram_mb
def run_config(cfg, max_frames=None):

    video_path = preprocess_video(VIDEO_PATH)
    spine = SpinePoseEstimator(device="cpu")
    mp_pose = mediapipe.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)

    spine_kps_smoothed = None
    frame_idx = 0

    start = time.time()

    while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if frame_idx % cfg["interval"] == 0 and results.pose_landmarks:
            raw = _spine_pass(spine, frame, width, height, cfg, results)
            if raw is not None:
                if spine_kps_smoothed is None:
                    spine_kps_smoothed = raw.copy()
                else:
                    spine_kps_smoothed = (
                        EMA_ALPHA * raw + (1 - EMA_ALPHA) * spine_kps_smoothed
                    )

    elapsed = time.time() - start
    cap.release()
    pose.close()
    return elapsed, frame_idx


# ################ Ablation ###################
print("\n" + "=" * 62)
print("SECTION 1 — Ablation (deadlift, full video)")
print("=" * 62)

ablation_results = []
for cfg in TEST_CONFIGS:
    print(f"\n  Running: {cfg['name']} ...")
    t, frames = run_config(cfg)
    ablation_results.append((cfg["name"], t, frames / t))
    print(f"    Time: {t:.1f}s  |  FPS: {frames/t:.1f}")

baseline_t = ablation_results[0][1]
print("\n" + "-" * 55)
print(f"  {'Configuration':<32} {'Time':>7}  {'FPS':>5}  {'Reduction':>10}")
print("-" * 55)
for name, t, fps in ablation_results:
    red = (baseline_t - t) / baseline_t * 100
    bar = f"{red:+.1f}%" if red != 0 else "baseline"
    print(f"  {name:<32} {t:>6.1f}s  {fps:>5.1f}  {bar:>10}")
print("=" * 55)


######### length scaling#############
print("\n" + "=" * 62)
print("SECTION 2 — Scaling (all optimisations, varying duration)")
print("=" * 62)

cap_tmp = cv2.VideoCapture(preprocess_video(VIDEO_PATH))
src_fps = cap_tmp.get(cv2.CAP_PROP_FPS)
cap_tmp.release()

all_opt_cfg = TEST_CONFIGS[-1]
scale_results = []

for dur in SCALE_DURATIONS:
    max_f = int(dur * src_fps)
    print(f"\n  Duration: {dur}s  (≈{max_f} frames) ...")
    t, frames = run_config(all_opt_cfg, max_frames=max_f)
    scale_results.append((dur, t, frames / t))
    print(f"    Time: {t:.1f}s  |  FPS: {frames/t:.1f}")

print("\n" + "-" * 45)
print(f"  {'Duration':>10}  {'Proc. time':>11}  {'FPS':>5}")
print("-" * 45)
for dur, t, fps in scale_results:
    print(f"  {dur:>9}s  {t:>10.1f}s  {fps:>5.1f}")
print("=" * 45)

# processing time vs video duration grpah
durations = [r[0] for r in scale_results]
proc_times = [r[1] for r in scale_results]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    durations,
    proc_times,
    marker="o",
    linewidth=2,
    color="#2563EB",
    label="Processing time",
)
ax.plot(durations, durations, linestyle="--", color="#9CA3AF", label="Real-time (1×)")
ax.set_xlabel("Video duration (s)")
ax.set_ylabel("Processing time (s)")
ax.set_title(
    "Deadlift processing time vs video duration\n(all optimisations, CPU — Apple M2)"
)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(GRAPH_OUT, dpi=150)
print(f"\nGraph saved → {GRAPH_OUT}")
plt.close()
