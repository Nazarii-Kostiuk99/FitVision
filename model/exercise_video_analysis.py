import sys
import cv2
import mediapipe
import numpy as np
import os

from Utility.angle_calculations import calculate_angle
from exercises.squat_analysis import analyse_squat, get_initial_squat_rep_data
from exercises.pushup_analysis import analyse_pushup, get_initial_pushup_rep_data
from exercises.deadlift_analysis import (
    analyse_deadlift,
    get_initial_deadlift_rep_data,
    SPINE_IDS,
    SPINE_CONFIDENCE_THRESHOLD,
    EMA_ALPHA,
)
from Utility.video_utility import mov_to_mp4, preprocess_video
from feedback_generation.llm_feedback import generate_set_feedback

# only import spinepose if its installed, dont want it crashing for other exercises
try:
    from spinepose import SpinePoseEstimator

    SPINEPOSE_AVAILABLE = True
except ImportError:
    SPINEPOSE_AVAILABLE = False
    print(
        "WARNING: spinepose not installed — deadlift spine curvature will be skipped."
    )


def analyse_video(video_path, exercise_type):
    """
    Supported: squat, pushup, deadlift
    Exercise selected by user, parsed to backend + called by django
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

    # create spinepose estimator once, only use if its a deadlift
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

    # deadlift needs complexity=1 for accurate spine/hip tracking
    # pushup/squat are fine with 0 — ~2x faster, no meaningful accuracy loss
    pose = mp_pose.Pose(
        model_complexity=0 if exercise_type in ("pushup", "squat") else 1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    # only run spinepose during down phase, no point checking spine at lockout.
    # last result gets reused on skipped frames so the overlay doesnt flicker
    SPINEPOSE_INTERVAL = 1
    spine_keypoints = None
    spine_scores = None
    spine_kps_smoothed = None  # EMA-smoothed vertebra positions across frames

    while cap.isOpened():
        ret, frame = cap.read()

        # if end of video or no frame
        if not ret:
            break

        frame_idx += 1
        video_ts = (
            cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        )  # actual video time in seconds

        # rgb for mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # run spinepose only during down phase, every N frames +pass mediapipe's bbox for spinepose skips its own detector +:speed
        # frame is halved before inference and coords scaled back up: +:speed
        if (
            exercise_type == "deadlift"
            and spine_estimator is not None
            and state == "down"
            and frame_idx % SPINEPOSE_INTERVAL == 0
            and results.pose_landmarks
        ):
            lms = results.pose_landmarks.landmark
            xs = [lm.x * width for lm in lms]
            ys = [lm.y * height for lm in lms]
            x1 = max(0, int(min(xs)) - 200)
            y1 = max(0, int(min(ys)) - 200)
            x2 = min(width, int(max(xs)) + 200)
            y2 = min(height, int(max(ys)) + 200)

            small = cv2.resize(frame, (width // 2, height // 2))
            # scale bbox to half-res before passing in
            bbox = np.array([[x1 // 2, y1 // 2, x2 // 2, y2 // 2]], dtype=np.float32)
            kps, scs = spine_estimator(small, bboxes=bbox)
            if len(kps) > 0:
                raw_kps = kps[0].copy()
                raw_kps[:, 0] *= 2
                raw_kps[:, 1] *= 2

                # EMA smoothing on each vertebra so single bad frames dont cause spikes
                if spine_kps_smoothed is None:
                    spine_kps_smoothed = raw_kps.copy()
                else:
                    spine_kps_smoothed = (
                        EMA_ALPHA * raw_kps + (1 - EMA_ALPHA) * spine_kps_smoothed
                    )

                spine_keypoints = spine_kps_smoothed
                spine_scores = scs[0]
            else:
                spine_keypoints = None
                spine_scores = None

        # drawing joints
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            if exercise_type in ("deadlift", "pushup"):
                # only draw the visible side — drawing both causes the back leg to flicker
                _side = rep_data.get("side") or "right"
                _prefix = "LEFT" if _side == "left" else "RIGHT"
                _ids = {
                    "shoulder": getattr(
                        mp_pose.PoseLandmark, f"{_prefix}_SHOULDER"
                    ).value,
                    "elbow": getattr(mp_pose.PoseLandmark, f"{_prefix}_ELBOW").value,
                    "hip": getattr(mp_pose.PoseLandmark, f"{_prefix}_HIP").value,
                    "knee": getattr(mp_pose.PoseLandmark, f"{_prefix}_KNEE").value,
                    "ankle": getattr(mp_pose.PoseLandmark, f"{_prefix}_ANKLE").value,
                    "wrist": getattr(mp_pose.PoseLandmark, f"{_prefix}_WRIST").value,
                }
                _connections = [
                    ("shoulder", "hip"),
                    ("hip", "knee"),
                    ("knee", "ankle"),
                    ("shoulder", "elbow"),
                    ("elbow", "wrist"),
                ]
                _pts = {
                    name: (
                        int(landmarks[idx].x * width),
                        int(landmarks[idx].y * height),
                    )
                    for name, idx in _ids.items()
                }
                for a, b in _connections:
                    cv2.line(frame, _pts[a], _pts[b], (0, 0, 255), 2)
                for name, pt in _pts.items():
                    cv2.circle(frame, pt, 5, (255, 255, 255), -1)
            else:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(255, 255, 255), thickness=2, circle_radius=3
                    ),
                    mp_drawing.DrawingSpec(
                        color=(0, 0, 255), thickness=2, circle_radius=3
                    ),
                )

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
                    landmarks,
                    mp_pose,
                    reps,
                    state,
                    rep_data,
                    rep_feedback_blocks,
                    video_ts,
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
                # only draw when in down phase
                if (
                    spine_keypoints is not None
                    and spine_scores is not None
                    and state == "down"
                ):
                    if cur_spine_curv < 0.03:
                        spine_colour = (0, 255, 0)  # green — neutral
                    elif cur_spine_curv < 0.25:
                        spine_colour = (0, 165, 255)  # orange  slight rounding
                    else:
                        spine_colour = (0, 0, 255)  # red very rounedd

                    # compute back-offset direction from mediapipe shoulder→hip vector.
                    # rotating torso vec 90° gives the "side" direction — we pick
                    # the rotation that points toward the back of the body.
                    SPINE_BACK_OFFSET = 100
                    s_lm = landmarks[
                        getattr(
                            mp_pose.PoseLandmark,
                            f"{'LEFT' if rep_data.get('side') == 'left' else 'RIGHT'}_SHOULDER",
                        ).value
                    ]
                    h_lm = landmarks[
                        getattr(
                            mp_pose.PoseLandmark,
                            f"{'LEFT' if rep_data.get('side') == 'left' else 'RIGHT'}_HIP",
                        ).value
                    ]
                    tx = s_lm.x * width - h_lm.x * width
                    ty = s_lm.y * height - h_lm.y * height
                    t_len = max((tx**2 + ty**2) ** 0.5, 1e-6)
                    # two candidate perpendiculars pick the one pointing away from the wrist
                    perp1 = (-ty / t_len, tx / t_len)
                    perp2 = (ty / t_len, -tx / t_len)
                    w_lm = landmarks[
                        getattr(
                            mp_pose.PoseLandmark,
                            f"{'LEFT' if rep_data.get('side') == 'left' else 'RIGHT'}_WRIST",
                        ).value
                    ]
                    mid_x = (s_lm.x + h_lm.x) / 2 * width
                    mid_y = (s_lm.y + h_lm.y) / 2 * height
                    wrist_dx = w_lm.x * width - mid_x
                    wrist_dy = w_lm.y * height - mid_y
                    # back direction is opposite to wrist direction
                    back_dir = (
                        perp1
                        if (perp1[0] * wrist_dx + perp1[1] * wrist_dy) < 0
                        else perp2
                    )
                    ox = int(back_dir[0] * SPINE_BACK_OFFSET)
                    oy = int(back_dir[1] * SPINE_BACK_OFFSET)

                    spine_pts_px = []
                    for idx in SPINE_IDS:
                        if spine_scores[idx] >= SPINE_CONFIDENCE_THRESHOLD:
                            px = int(spine_keypoints[idx][0]) + ox
                            py = int(spine_keypoints[idx][1]) + oy
                            spine_pts_px.append((px, py))
                            cv2.circle(frame, (px, py), 5, spine_colour, -1)
                        else:
                            spine_pts_px.append(None)

                    # connect  vertebras with lines
                    for i in range(len(spine_pts_px) - 1):
                        if spine_pts_px[i] and spine_pts_px[i + 1]:
                            cv2.line(
                                frame,
                                spine_pts_px[i],
                                spine_pts_px[i + 1],
                                spine_colour,
                                2,
                            )

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

        # one LLM call for the whole set after all reps are printed
        print("\n------------------------------------------------------------")
        print("  LLM SUMMARY")
        print("------------------------------------------------------------")
        llm_summary = generate_set_feedback(exercise_type, rep_feedback_blocks)
        print(f"\n{llm_summary}")


# entry point
if __name__ == "__main__":
    if len(sys.argv) > 2:
        video_file = sys.argv[1]
        exercise = sys.argv[2]
        analyse_video(video_file, exercise)
    else:
        # defaults to sqaut
        analyse_video("test_squat.mov", "squat")
