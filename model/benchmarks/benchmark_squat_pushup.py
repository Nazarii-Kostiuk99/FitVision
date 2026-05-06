# benchmark_squat_pushup.py
# benchmark_squat_pushup.py

# venv/bin/python3 model/benchmarks/benchmark_squat_pushup.py squat   /Users/kostiuk_nazarii/Desktop/project_videos/in/squats/shallow_squats.mov
#  venv/bin/python3 model/benchmarks/benchmark_squat_pushup.py pushup /Users/kostiuk_nazarii/Desktop/project_videos/in/pushups/p_GOOD.mov

import time
import cv2
import mediapipe
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from Utility.video_utility import preprocess_video
from exercises.squat_analysis import analyse_squat, get_initial_squat_rep_data
from exercises.pushup_analysis import analyse_pushup, get_initial_pushup_rep_data

VIDEOS = {
    "squat": "/Users/kostiuk_nazarii/Desktop/project_videos/in/squats/benchmark/good_squats.mov",  # 15 sec
    "pushup": "/Users/kostiuk_nazarii/Desktop/project_videos/in/pushups/benchmark/p_GOOD.mov",  # 15 sec
}


def run(video_path, exercise):
    video_path = preprocess_video(video_path)

    mp_pose = mediapipe.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=0,  # matches production pipeline for squat/pushup
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / src_fps if src_fps > 0 else 0

    reps = 0
    state = "up"
    rep_data = (
        get_initial_squat_rep_data()
        if exercise == "squat"
        else get_initial_pushup_rep_data()
    )
    feedback = []
    max_depth_ratio = 0.0
    min_elbow_angle = 999.0

    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            if exercise == "squat":
                out = analyse_squat(lms, mp_pose, reps, state, rep_data, feedback)
                max_depth_ratio = max(max_depth_ratio, out[5])
                reps, state, rep_data, feedback = out[6], out[7], out[8], out[9]
            else:
                out = analyse_pushup(
                    lms, mp_pose, reps, state, rep_data, feedback, video_timestamp=ts
                )
                val = out[5]
                if val < 999:
                    min_elbow_angle = min(min_elbow_angle, val)
                reps, state, rep_data, feedback = out[6], out[7], out[8], out[9]

    elapsed = time.time() - start
    cap.release()
    pose.close()
    return elapsed, total_frames, duration, max_depth_ratio, min_elbow_angle


print("=" * 62)
print("Squat / Push-up benchmark  (MediaPipe + analysis modules, CPU)")
print("=" * 62)

# usage: python benchmark_squat_pushup.py squat /path/to/video.mov
if len(sys.argv) == 3:
    to_run = {sys.argv[1]: sys.argv[2]}
else:
    to_run = VIDEOS

print(f"\n  {'Exercise':<10} {'Duration':>9}  {'Proc. time':>11}  {'FPS':>5}")
print("-" * 50)

for exercise, path in to_run.items():
    print(f"\n  Running {exercise} ...")
    t, frames, dur, max_depth, min_elbow = run(path, exercise)
    fps = frames / t
    print(f"  {exercise:<10} {dur:>8.1f}s  {t:>10.1f}s  {fps:>5.1f}")
    if exercise == "squat":
        print(
            f"  peak depth_ratio: {max_depth:.3f}  (shallow <0.70, good 0.70–1.20, too deep >1.20)"
        )
    if exercise == "pushup":
        print(
            f"  min elbow_angle: {min_elbow:.1f}°  (excellent <80°, good 80–95°, shallow >95°)"
        )

print("\n" + "=" * 62)
