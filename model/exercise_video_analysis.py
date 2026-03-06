import sys
import cv2
import mediapipe
import numpy as np
import os
from Utility.angle_calculations import calculate_angle
from exercises.squat_analysis import analyse_squat, get_initial_squat_rep_data
from exercises.pushup_analysis import (
    analyse_pushup,
    get_initial_pushup_rep_data,
)  # Add this
from Utility.video_utility import mov_to_mp4, preprocess_video

# analyse_pushup, analyse_lunge, analyse_plank


def analyse_video(video_path, exercise_type):
    """
    Supported: squat, pushup, lunge, plank
    Exercise selected by user -> parsed to backend + called by django
    """

    if exercise_type not in ["squat", "pushup", "lunge", "plank"]:
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
    else:
        rep_data = {}

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

    pose = mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )  # test other value mb better accuracy

    while cap.isOpened():
        ret, frame = cap.read()

        # if end of video or no frame
        if not ret:
            break

        # rgb for mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

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
