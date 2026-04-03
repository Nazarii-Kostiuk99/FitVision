import sys
import cv2
import mediapipe
import numpy as np
import os
import torch
from tqdm import tqdm

from exercises.squat_analysis import analyse_squat, get_initial_squat_rep_data
from exercises.pushup_analysis import analyse_pushup, get_initial_pushup_rep_data
from exercises.deadlift_analysis import (
    analyse_deadlift,
    get_initial_deadlift_rep_data,
    SPINE_IDS,
    SPINE_CONFIDENCE_THRESHOLD,
    EMA_ALPHA,
)
from Utility.video_utility import preprocess_video
from feedback_generation.llm_feedback import generate_set_feedback

#  CUDA (my 1660), MPS (Apple M2), CPU fallback
if torch.cuda.is_available():
    TORCH_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    TORCH_DEVICE = "mps"
else:
    TORCH_DEVICE = "cpu"

print(f"Device: {TORCH_DEVICE}")

# only import spinepose if installed — deadlift only, don't crash other exercises
try:
    from spinepose import SpinePoseEstimator

    SPINEPOSE_AVAILABLE = True
except ImportError:
    SPINEPOSE_AVAILABLE = False
    print(
        "WARNING: spinepose not installed — deadlift spine curvature will be skipped."
    )


def analyse_video(video_path, exercise_type, output_dir="outputs", analysis_id=None):
    """
    Supported: squat, pushup, deadlift
    Exercise selected by user, parsed to backend + called by django
    Returns a dict with rep_feedback, llm_summary, total_reps, output_video_path
    """

    if exercise_type not in ["squat", "pushup", "deadlift", "lunge", "plank"]:
        print(f"Exercise '{exercise_type}' not supported yet")
        return None

    base_name = f"analysis_{analysis_id}" if analysis_id else os.path.splitext(os.path.basename(video_path))[0]

    video_path = preprocess_video(video_path)  # mov (iPhone format) -> anonymised mp4
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

    # create spinepose estimator once, only used for deadlift
    spine_estimator = None
    if exercise_type == "deadlift" and SPINEPOSE_AVAILABLE:
        spine_estimator = SpinePoseEstimator(device=TORCH_DEVICE)

    rep_feedback_blocks = []

    mp_pose = mediapipe.solutions.pose
    mp_drawing = mediapipe.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # deadlift needs complexity=1 for accurate spine/hip tracking
    # pushup/squat fine with 0 — much faster, no meaningful accuracy loss
    pose = mp_pose.Pose(
        model_complexity=0 if exercise_type in ("pushup", "squat") else 1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    # only run spinepose during down phase — last result reused on skipped frames so overlay doesn't flicker
    SPINEPOSE_INTERVAL = 2
    spine_keypoints = None
    spine_scores = None
    spine_kps_smoothed = None  # EMA-smoothed vertebra positions across frames

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Processing {exercise_type}", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        pbar.update(1)
        video_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # run spinepose only during down phase, every N frames
        # pass mediapipe bbox to skip spinepose's own detector (+speed)
        # frame halved before inference, coords scaled back up (+speed)
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
            x1 = max(0, int(min(xs)) - 150)
            y1 = max(0, int(min(ys)) - 150)
            x2 = min(width, int(max(xs)) + 150)
            y2 = min(height, int(max(ys)) + 150)

            small = cv2.resize(frame, (width // 2, height // 2))
            bbox = np.array([[x1 // 2, y1 // 2, x2 // 2, y2 // 2]], dtype=np.float32)
            kps, scs = spine_estimator(small, bboxes=bbox)
            if len(kps) > 0:
                raw_kps = kps[0].copy()
                raw_kps[:, 0] *= 2
                raw_kps[:, 1] *= 2

                # EMA smoothing per vertebra — prevents single bad frames causing spikes
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

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            if exercise_type in ("deadlift", "pushup"):
                # only draw the visible side — drawing both causes the back limbs to flicker
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
                for pt in _pts.values():
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

            if exercise_type == "squat":
                (
                    angle,
                    feedback,
                    color,
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

                GOOD = (80, 200, 80)
                WARN = (0, 165, 255)
                BAD = (60, 60, 220)
                WHITE = (240, 240, 240)
                GOLD = (50, 200, 220)
                metrics = [
                    (
                        "Knee Angle  (bend at bottom)",
                        f"{int(angle)} deg",
                        GOOD if angle < 100 else WARN,
                    ),
                    (
                        "Squat Depth  (parallel = 1.0)",
                        f"{depth_ratio:.2f}",
                        GOOD if depth_ratio >= 0.9 else WARN,
                    ),
                    (
                        "Torso Lean  (ideally < 45 deg)",
                        f"{lean_angle:.1f} deg",
                        GOOD if lean_angle < 45 else WARN,
                    ),
                    (
                        "Heels",
                        "Lifted" if heel_lifted_current else "Grounded",
                        BAD if heel_lifted_current else GOOD,
                    ),
                ]

            if exercise_type == "pushup":
                (
                    angle,
                    feedback,
                    color,
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

                GOOD = (80, 200, 80)
                WARN = (0, 165, 255)
                BAD = (60, 60, 220)
                WHITE = (240, 240, 240)
                GOLD = (50, 200, 220)
                metrics = [
                    (
                        "Elbow Angle  (arm bend at bottom)",
                        f"{int(angle)} deg",
                        GOOD if angle < 100 else WARN,
                    ),
                    (
                        "Body Line  (straight = 180 deg)",
                        f"{int(body_angle)} deg",
                        GOOD if abs(body_angle - 180) < 15 else WARN,
                    ),
                    (
                        "Shoulder Shift  (lateral drift)",
                        f"{shoulder_offset_current:.2f}",
                        GOOD if shoulder_offset_current < 0.05 else WARN,
                    ),
                    (
                        "Neck Angle  (head position)",
                        f"{int(neck_angle)} deg",
                        GOOD if neck_angle < 30 else WARN,
                    ),
                ]

            if exercise_type == "deadlift":
                (
                    angle,
                    feedback,
                    color,
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

                # spine overlay — only during down phase
                if (
                    spine_keypoints is not None
                    and spine_scores is not None
                    and state == "down"
                ):
                    if cur_spine_curv < 0.03:
                        spine_colour = (0, 255, 0)  # green — neutral
                    elif cur_spine_curv < 0.25:
                        spine_colour = (0, 165, 255)  # orange — slight rounding
                    else:
                        spine_colour = (0, 0, 255)  # red — excessive rounding

                    # offset spine overlay toward the back of the body
                    # rotate torso vec (shoulder→hip) 90° and pick direction away from wrist
                    SPINE_BACK_OFFSET = 100
                    side_key = "LEFT" if rep_data.get("side") == "left" else "RIGHT"
                    s_lm = landmarks[
                        getattr(mp_pose.PoseLandmark, f"{side_key}_SHOULDER").value
                    ]
                    h_lm = landmarks[
                        getattr(mp_pose.PoseLandmark, f"{side_key}_HIP").value
                    ]
                    w_lm = landmarks[
                        getattr(mp_pose.PoseLandmark, f"{side_key}_WRIST").value
                    ]

                    tx = s_lm.x * width - h_lm.x * width
                    ty = s_lm.y * height - h_lm.y * height
                    t_len = max((tx**2 + ty**2) ** 0.5, 1e-6)
                    perp1 = (-ty / t_len, tx / t_len)
                    perp2 = (ty / t_len, -tx / t_len)

                    mid_x = (s_lm.x + h_lm.x) / 2 * width
                    mid_y = (s_lm.y + h_lm.y) / 2 * height
                    wrist_dx = w_lm.x * width - mid_x
                    wrist_dy = w_lm.y * height - mid_y
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

                    for i in range(len(spine_pts_px) - 1):
                        if spine_pts_px[i] and spine_pts_px[i + 1]:
                            cv2.line(
                                frame,
                                spine_pts_px[i],
                                spine_pts_px[i + 1],
                                spine_colour,
                                2,
                            )

                GOOD = (80, 200, 80)
                WARN = (0, 165, 255)
                BAD = (60, 60, 220)
                WHITE = (240, 240, 240)
                GOLD = (50, 200, 220)
                metrics = [
                    (
                        "Hip Angle  (hinge depth)",
                        f"{int(angle)} deg",
                        GOOD if angle < 90 else WARN,
                    ),
                    (
                        "Back Angle  (torso from vertical)",
                        f"{int(back_angle)} deg",
                        GOOD if back_angle < 40 else WARN,
                    ),
                    (
                        "Spine Curve  (0 = neutral spine)",
                        f"{cur_spine_curv:.3f}",
                        GOOD if cur_spine_curv < 0.03 else (WARN if cur_spine_curv < 0.25 else BAD),
                    ),
                    (
                        "Bar Drift  (distance from body)",
                        f"{cur_bar_drift:.3f}",
                        GOOD if cur_bar_drift < 0.05 else WARN,
                    ),
                    ("Knee Angle", f"{int(knee_angle_cur)} deg", WHITE),
                ]

            # ── overlay helpers ───────────────────────────────────────────
            def draw_panel(f, x1, y1, x2, y2, alpha=0.55):
                overlay = f.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 15), -1)
                cv2.addWeighted(overlay, alpha, f, 1 - alpha, 0, f)
                cv2.rectangle(f, (x1, y1), (x2, y2), (60, 60, 60), 1)

            def draw_metric(f, x, y, label, value, val_color=(240, 240, 240)):
                cv2.putText(f, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (190, 190, 190), 1, cv2.LINE_AA)
                cv2.putText(f, value, (x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                            val_color, 2, cv2.LINE_AA)

            # ── metrics panel (top-right) ─────────────────────────────────
            panel_w = 360
            row_h = 50
            pad = 10
            panel_h = pad + len(metrics) * row_h + pad
            px1 = width - panel_w - 10
            py1 = 10
            draw_panel(frame, px1, py1, width - 10, py1 + panel_h)
            for i, (label, value, val_color) in enumerate(metrics):
                row_y = py1 + pad + i * row_h
                draw_metric(frame, px1 + 10, row_y + 14, label, value, val_color)

            # ── exercise title + rep counter + phase (top-left) ──────────
            caption = f"{exercise_type.capitalize()} analysis"
            (cw, _ch), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
            title_panel_w = max(cw + 20, 300)
            draw_panel(frame, 10, 10, title_panel_w, 145)
            cv2.putText(frame, caption, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (180, 180, 180), 1, cv2.LINE_AA)
            # rep number — shrink font if >= 10 to avoid overflow
            rep_str = str(reps)
            rep_scale = 2.6
            rep_y = 118
            cv2.putText(frame, rep_str, (18, rep_y), cv2.FONT_HERSHEY_DUPLEX,
                        rep_scale, GOLD, 3, cv2.LINE_AA)
            rep_txt_w = 140
            cv2.putText(frame, "REPS", (rep_txt_w, 68), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (150, 150, 150), 1, cv2.LINE_AA)
            phase_col = GOOD if state == "up" else WARN
            cv2.putText(frame, state.upper(), (rep_txt_w, 96), cv2.FONT_HERSHEY_SIMPLEX,
                        0.78, phase_col, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{side.capitalize()} side", (rep_txt_w, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 130, 130), 1, cv2.LINE_AA)

            # ── feedback bar (only shown when there is a message) ─────────
            if feedback:
                (fw, _fh), _ = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
                draw_panel(frame, 10, 152, fw + 30, 192)
                cv2.putText(frame, feedback, (18, 180), cv2.FONT_HERSHEY_SIMPLEX,
                            0.85, color, 2, cv2.LINE_AA)

            # ── joint angle labels ────────────────────────────────────────
            for _name, (ang, coord) in angles.items():
                x = int(coord[0] * width)
                y = int(coord[1] * height)
                label = f"{int(ang)} deg"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                cv2.rectangle(frame, (x + 6, y - th - 16), (x + tw + 14, y - 4),
                              (20, 20, 20), -1)
                cv2.putText(frame, label, (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, WHITE, 1, cv2.LINE_AA)

        out.write(frame)

    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # encoding back to to h-264 so that ios can stream it, mp4v crahing
    h264_filename = output_filename.replace(".mp4", "_h264.mp4")
    os.system(
        f'ffmpeg -y -i "{output_filename}" -vcodec libx264 -acodec aac "{h264_filename}" -loglevel quiet'
    )
    if os.path.exists(h264_filename):
        os.replace(h264_filename, output_filename)

    # delete the anonymised intermediate — only the final annotated output is needed
    if os.path.exists(video_path):
        os.remove(video_path)

    print("\n============================================================")
    print("                 SET FEEDBACK (PER REP)")
    print("============================================================")

    llm_summary = ""
    if len(rep_feedback_blocks) == 0:
        print("No reps detected.")
    else:
        for block in rep_feedback_blocks:
            print(f"\n### REP {block['rep']} ({block['side']}) ###")
            for line in block["feedback"]:
                print(f" - {line}")

        print("\n------------------------------------------------------------")
        print("  LLM SUMMARY")
        print("------------------------------------------------------------")
        llm_summary = generate_set_feedback(exercise_type, rep_feedback_blocks)
        print(f"\n{llm_summary}")

    return {
        "total_reps": reps,
        "rep_feedback": rep_feedback_blocks,
        "llm_summary": llm_summary,
        "output_video_path": output_filename,
    }


if __name__ == "__main__":
    if len(sys.argv) > 2:
        analyse_video(sys.argv[1], sys.argv[2])
    else:
        print("python exercise_video_analysis.py <video_path> <exercise>")
        print("Exercises: squat, pushup, deadlift")
