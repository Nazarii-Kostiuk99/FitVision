import sys
import cv2
import mediapipe
import numpy as np
import os
from Utility.angle_calculations import calculate_angle
from exercises.squat_analysis import analyse_squat, get_initial_squat_rep_data
from exercises.pushup_analysis import analyse_pushup, get_initial_pushup_rep_data
from exercises.deadlift_analysis import analyse_deadlift, get_initial_deadlift_rep_data, SPINE_IDS, SPINE_CONFIDENCE_THRESHOLD
from Utility.video_utility import mov_to_mp4, preprocess_video

# SpinePose is only imported when needed (deadlift) to keep startup fast
# for squat/pushup sessions that don't use it.
try:
    from spinepose import SpinePoseEstimator
    SPINEPOSE_AVAILABLE = True
except ImportError:
    SPINEPOSE_AVAILABLE = False
    print("WARNING: spinepose not installed — deadlift spine curvature will be skipped."
          "  Install with: pip install spinepose")


def analyse_video(video_path, exercise_type):
    """
    Supported: squat, pushup, lunge, plank
    Exercise selected by user -> parsed to backend + called by django
    """

    if exercise_type not in ["squat", "pushup", "deadlift", "lunge", "plank"]:
        print(f"Exercise '{exercise_type}' not supported yet")
        return

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    video_path = preprocess_video(video_path)  # mov(iphone format) -> mp4
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"{base_name}_analysis.mp4")

    reps = 0
    state = "up"

    if exercise_type == "squat":
        rep_data = get_initial_squat_rep_data()
    elif exercise_type == "pushup":
        rep_data = get_initial_pushup_rep_data()
    elif exercise_type == "deadlift":
        rep_data = get_initial_deadlift_rep_data()
    else:
        rep_data = {}

    # SpinePose estimator — instantiated once here, used every frame for deadlift.
    # Uses CPU by default; change to 'cuda' if a GPU is available.
    spine_estimator = None
    if exercise_type == "deadlift" and SPINEPOSE_AVAILABLE:
        spine_estimator = SpinePoseEstimator(device="cpu")

    rep_feedback_blocks = []

    mp_pose = mediapipe.solutions.pose
    mp_drawing = mediapipe.solutions.drawing_utils

    # opencv reads a video from a file
    cap = cv2.VideoCapture(video_path)

    # record fps, dimensions of input video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # writer output logic
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # PERF_TEST: model_complexity=0 (lite) vs 1 (full, default).
    # Lite is ~2x faster with negligible accuracy loss for large landmarks (hip/knee/shoulder).
    # To revert to full model: change 0 → 1
    pose = mp_pose.Pose(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    # SpinePose is run every SPINEPOSE_INTERVAL frames and only during the down
    # phase — curvature at lockout is irrelevant and skipping "up" frames alone
    # halves the number of inference calls. The last result is reused on skipped frames.
    SPINEPOSE_INTERVAL = 3
    spine_keypoints = None
    spine_scores = None

    while cap.isOpened():
        ret, frame = cap.read()

        # if end of video or no frame
        if not ret:
            break

        frame_idx += 1
        video_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # actual video time in seconds

        # rgb for mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # SpinePose inference — deadlift only, down phase only, every N frames.
        # Frame is halved before inference and pixel coords scaled back up after,
        # reducing inference time without meaningful accuracy loss for a side-view
        # subject filling most of the frame. Last result reused on skipped frames.
        if (
            exercise_type == "deadlift"
            and spine_estimator is not None
            and state == "down"
            and frame_idx % SPINEPOSE_INTERVAL == 0
        ):
            small = cv2.resize(frame, (width // 2, height // 2))
            kps, scs = spine_estimator(small)
            if len(kps) > 0:
                spine_keypoints = kps[0].copy()
                spine_keypoints[:, 0] *= 2  # scale x back to full resolution
                spine_keypoints[:, 1] *= 2  # scale y back to full resolution
                spine_scores = scs[0]
            else:
                spine_keypoints = None
                spine_scores = None

        # drawing points of interest (joints)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2, circle_radius=3
                ),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
            )

            landmarks = results.pose_landmarks.landmark

            # call the appropriate exercise type analyser
            if exercise_type == "squat":
                (
                    summary,
                    angle,
                    feedback,
                    color,
                    label,
                    side,
                    angles,
                    depth_ratio,
                    reps,
                    state,
                    rep_data,
                    rep_feedback_blocks,
                    heel_lifted_current,
                    lean_angle,
                ) = analyse_squat(
                    landmarks, mp_pose, reps, state, rep_data, rep_feedback_blocks
                )

                cv2.rectangle(
                    frame, (width - 320, 10), (width - 10, 260), (0, 0, 0), -1
                )
                debug_stats = [
                    (f"State: {state}", (255, 0, 255)),
                    (f"CUR Depth: {depth_ratio:.2f}", (0, 255, 0)),
                    (f"MAX Depth: {rep_data['max_depth_ratio']:.2f}", (0, 255, 255)),
                    (f"CUR Knee: {int(angle)}", (0, 255, 0)),
                    (f"MIN Knee: {int(rep_data['min_knee_angle'])}", (0, 255, 255)),
                    (f"Lean (deg): {lean_angle:.1f}", (0, 255, 255)),
                    (f"Heel Lift: {heel_lifted_current}", (0, 255, 255)),
                ]

            if exercise_type == "pushup":
                (
                    summary,
                    angle,
                    feedback,
                    color,
                    label,
                    side,
                    angles,
                    depth_ratio,
                    reps,
                    state,
                    rep_data,
                    rep_feedback_blocks,
                    shoulder_offset_current,
                    body_angle,
                    neck_angle,
                    cur_min_body,
                    cur_max_body,
                ) = analyse_pushup(
                    landmarks, mp_pose, reps, state, rep_data, rep_feedback_blocks
                )

                debug_stats = [
                    (f"State: {state}", (255, 0, 255)),
                    (f"CUR Elbow: {int(angle)}", (0, 255, 0)),
                    (f"MIN Elbow: {int(rep_data['min_elbow_angle'])}", (0, 255, 255)),
                    (f"Body Ang: {int(body_angle)}", (0, 255, 255)),
                    (f"Min Body: {int(cur_min_body)}", (0, 255, 0)),
                    (f"Max Body: {int(cur_max_body)}", (0, 255, 0)),
                    (f"S-Offset: {shoulder_offset_current:.2f}", (0, 255, 255)),
                    (f"CUR Neck: {int(neck_angle)}", (0, 255, 0)),
                ]

            if exercise_type == "deadlift":
                (
                    summary,
                    angle,
                    feedback,
                    color,
                    label,
                    side,
                    angles,
                    knee_angle_cur,
                    reps,
                    state,
                    rep_data,
                    rep_feedback_blocks,
                    back_angle,
                    knee_angle_cur,
                    cur_min_hip,
                    cur_spine_curv,
                    cur_bar_drift,
                ) = analyse_deadlift(
                    landmarks,
                    mp_pose,
                    spine_keypoints,
                    spine_scores,
                    width,
                    height,
                    reps,
                    state,
                    rep_data,
                    rep_feedback_blocks,
                    video_ts,
                )

                # --- SPINE OVERLAY ---
                # Draw SpinePose vertebrae as dots connected by a line.
                # Colour reflects current curvature: green=neutral, yellow=slight, red=significant.
                if spine_keypoints is not None and spine_scores is not None:
                    if cur_spine_curv < 0.05:
                        spine_colour = (0, 255, 0)    # green — neutral
                    elif cur_spine_curv < 0.10:
                        spine_colour = (0, 165, 255)  # orange — slight rounding
                    else:
                        spine_colour = (0, 0, 255)    # red — significant rounding

                    spine_pts_px = []
                    for idx in SPINE_IDS:
                        if spine_scores[idx] >= SPINE_CONFIDENCE_THRESHOLD:
                            px = int(spine_keypoints[idx][0])
                            py = int(spine_keypoints[idx][1])
                            spine_pts_px.append((px, py))
                            cv2.circle(frame, (px, py), 5, spine_colour, -1)
                        else:
                            spine_pts_px.append(None)

                    # connect consecutive confident vertebrae with lines
                    for i in range(len(spine_pts_px) - 1):
                        if spine_pts_px[i] and spine_pts_px[i + 1]:
                            cv2.line(frame, spine_pts_px[i], spine_pts_px[i + 1], spine_colour, 2)

                debug_stats = [
                    (f"State: {state}", (255, 0, 255)),
                    (f"CUR Hip: {int(angle)}", (0, 255, 0)),
                    (f"MIN Hip: {int(cur_min_hip)}", (0, 255, 255)),
                    (f"Back Ang: {int(back_angle)}", (0, 255, 255)),
                    (f"Spine Curv: {cur_spine_curv:.3f}", (0, 165, 255)),
                    (f"Bar Drift: {cur_bar_drift:.3f}", (0, 255, 255)),
                    (f"CUR Knee: {int(knee_angle_cur)}", (0, 255, 0)),
                ]

            # --- DEBUG PANEL ---

            for i, (text, col) in enumerate(debug_stats):
                cv2.putText(
                    frame,
                    text,
                    (width - 300, 40 + (i * 35)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    col,
                    2,
                )

            # --- MAIN DISPLAY (Left Side) ---
            cv2.putText(
                frame,
                f"REPS: {reps}",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                3,
            )

            cv2.putText(
                frame, feedback, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
            )
            cv2.putText(
                frame,
                f"Side: {side}",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            # drawing angles for corresponding joints
            for name, (ang, coord) in angles.items():
                x = int(coord[0] * width)
                y = int(coord[1] * height)
                cv2.putText(
                    frame,
                    f"{int(ang)}",
                    (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        # write output video to a separate file (later send to django api -> DB)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # PRINT FEEDBACK ONCE PER REP (end of video)
    print("\n============================================================")
    print("                 SET FEEDBACK (PER REP)")
    print("============================================================")

    if len(rep_feedback_blocks) == 0:
        print("No reps detected.")
    else:
        for block in rep_feedback_blocks:
            print(f"\n### REP {block['rep']} ({block['side']}) ###")
            for line in block["feedback"]:
                print(f" - {line}")


# entry point
if __name__ == "__main__":
    if len(sys.argv) > 2:
        video_file = sys.argv[1]
        exercise = sys.argv[2]
        analyse_video(video_file, exercise)
    else:
        # defaults to sqaut
        analyse_video("test_squat.mov", "squat")
