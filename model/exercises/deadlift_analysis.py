"""
Deadlift

Counter = hip hinge angle (shoulder → hip → knee)
Spine curve --> SpinePose framework based on additional 9-points (perpendicular-deviation method)
Key metrics: bar path, knee lockout, tempo

SpinePose keypoint indices (top --> bottom):
    36 = neck_03   C1          (top of cervical spine)
    35  neck_02   C3/C4
    18  neck      C7          (cervicothoracic junction)
    30  spine_05  T3          (upper thoracic)
    29  spine_04  T8          (mid thoracic)
    28  spine_03  T12/L1      (thoracolumbar junction)
    27  spine_02  L3
    26  spine_01  L5
    19  hip       sacrum      (base of spine)
"""

from Utility.angle_calculations import calculate_angle, angle_to_vertical_degrees
import numpy as np

# SpinePose indices anatomical order (C1 → sacrum)
SPINE_IDS = [36, 35, 18, 30, 29, 28, 27, 26, 19]

# Minimum confidence score to trust a SpinePose keypoint
SPINE_CONFIDENCE_THRESHOLD = 0.3

# Minimum MediaPipe landmark visibility to trust for state machine updates.
# Frames where any key landmark falls below this are skipped — they don't
# advance the state machine or update accumulators (display only).
LANDMARK_VISIBILITY_THRESHOLD = 0.5

# Exponential moving average smoothing factor for landmark positions.
# Higher = more responsive but jitterier; lower = smoother but laggier.
EMA_ALPHA = 0.4


# per-rep tracking data ->  Reset after every completed rep
def get_initial_deadlift_rep_data():
    return {
        "side": None,
        "min_hip_angle": 999,  # hinge depth
        "spine_curvatures": [],  # normalised curvature value during each down state frame
        "wrist_x_positions": [],  # wrist x for bar path
        "knee_angle_at_top": None,  # knee angle on down --> up transition (lockout)
        "rep_start_time": None,
        "rep_end_time": None,
        "valid": False,  # True after hip_angle < 90
        # EMA-smoothed landmark positions — persisted across frames to reduce jitter
        "smooth_shoulder": None,
        "smooth_hip": None,
        "smooth_knee": None,
        "smooth_ankle": None,
        "smooth_wrist": None,
    }


################################ SIDE DETECTION ###############################


def detect_side(landmarks, mp_pose):
    left_points = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
    ]

    right_points = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]

    left_visibility = np.mean([landmarks[i].visibility for i in left_points])

    right_visibility = np.mean([landmarks[i].visibility for i in right_points])

    if left_visibility >= right_visibility:
        return "left"
    else:
        return "right"


################################ SPINE CURVATURE (SP) ###############################


# spine_points == list of (x, y)  top --> bottom
# 0.0 = perfectly straight, larger values = more curved
def compute_spine_curvature(spine_points):
    """
    Returns
    -------
    float


    How it works
    ------------
    1.  Draw the straight reference line from C1 (top) to sacrum (bottom).
    2.  For every intermediate point, compute its perpendicular distance to
        that line using the 2-D cross-product formula:
            distance = |AB × AP| / |AB|
        where AB = sacrum - C1, AP = point - C1.
    3.  Take the maximum distance across all intermediate vertebrae.
    4.  Divide by spine length so the result is scale-independent (works
        regardless of how close the camera is to the subject).
    """
    pts = np.array(spine_points, dtype=float)
    top = pts[0]  # C1
    bottom = pts[-1]  # sacrum

    spine_vec = bottom - top  # AB
    spine_len = np.linalg.norm(spine_vec)

    if spine_len < 1e-6:
        return 0.0

    max_dev = 0.0
    for pt in pts[1:-1]:  # intermediate vertebrae only
        ap = pt - top  # AP
        # 2-D cross product magnitude = |AB_x * AP_y  -  AB_y * AP_x|
        cross = abs(spine_vec[0] * ap[1] - spine_vec[1] * ap[0])
        perp_dist = cross / spine_len
        max_dev = max(max_dev, perp_dist)

    return (
        max_dev / spine_len
    )  # max perpendicular deviation of  vertebra from the C1-to-sacrum straight line, normalised by spine length.


def _extract_spine_points(spine_keypoints, spine_scores, frame_width, frame_height):
    """

    SpinePose returns keypoints in pixel coordinates; we normalise to [0, 1]
    so the curvature value is consistent with the rest of the pipeline which
    works in MediaPipe's normalised coordinate space.

    Returns None if any required keypoint falls below the confidence threshold,
    so that unreliable frames are silently skipped rather than corrupting the
    curvature accumulator.
    """
    points = []
    for idx in SPINE_IDS:
        if spine_scores[idx] < SPINE_CONFIDENCE_THRESHOLD:
            return None  # drop the whole frame if any point is uncertain
        x_norm = spine_keypoints[idx][0] / frame_width
        y_norm = spine_keypoints[idx][1] / frame_height
        points.append([x_norm, y_norm])
    return points


################################ MAIN ANALYSER ###############################


def analyse_deadlift(
    landmarks,
    mp_pose,
    spine_keypoints,  # np.ndarray shape (37, 2) pixel coords, or None
    spine_scores,  # np.ndarray shape (37,)  confidence,   or None
    frame_width,
    frame_height,
    reps,
    state,
    rep_data,
    rep_feedback_blocks,
    video_timestamp=0.0,  # seconds
):
    """
    Called once per frame from the main video loop.

    MediaPipe = state machine, bar-path tracking, knee-lockout check, timing.
    SpinePose = spine curvature metric. If SpinePose returns None for a frame
    the curvature accumulator simply receives no entry for that frame, and
    the mean is taken over whatever frames were reliable.
    """

    # settings side after detection
    if rep_data["side"] is None:
        rep_data["side"] = detect_side(landmarks, mp_pose)

    side_detected = rep_data["side"]

    if side_detected == "left":
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        side_label = "Left"
    else:
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        side_label = "Right"

    # ── visibility gate ─────────────────────────────────────────────────────────
    # Check reliability BEFORE EMA so we don't feed bad positions into the smoother.
    landmarks_reliable = all(
        lm.visibility >= LANDMARK_VISIBILITY_THRESHOLD
        for lm in [shoulder, hip, knee, ankle]
    )

    # ── EMA smoothing ──────────────────────────────────────────────────────────
    # Only update smoothed positions on reliable frames. On occluded frames the
    # smoother is frozen at the last good value — preventing the barbell plate
    # from gradually pulling landmark positions to wrong locations.
    def _ema(prev, new_val):
        if prev is None:
            return new_val
        return [EMA_ALPHA * n + (1 - EMA_ALPHA) * p for p, n in zip(prev, new_val)]

    if landmarks_reliable:
        rep_data["smooth_shoulder"] = _ema(
            rep_data["smooth_shoulder"], [shoulder.x, shoulder.y]
        )
        rep_data["smooth_hip"] = _ema(rep_data["smooth_hip"], [hip.x, hip.y])
        rep_data["smooth_knee"] = _ema(rep_data["smooth_knee"], [knee.x, knee.y])
        rep_data["smooth_ankle"] = _ema(rep_data["smooth_ankle"], [ankle.x, ankle.y])
        rep_data["smooth_wrist"] = _ema(rep_data["smooth_wrist"], [wrist.x, wrist.y])
    elif rep_data["smooth_shoulder"] is None:
        # First frame is unreliable — seed with raw values so coords are never None
        rep_data["smooth_shoulder"] = [shoulder.x, shoulder.y]
        rep_data["smooth_hip"] = [hip.x, hip.y]
        rep_data["smooth_knee"] = [knee.x, knee.y]
        rep_data["smooth_ankle"] = [ankle.x, ankle.y]
        rep_data["smooth_wrist"] = [wrist.x, wrist.y]

    shoulder_coords = rep_data["smooth_shoulder"]
    hip_coords = rep_data["smooth_hip"]
    knee_coords = rep_data["smooth_knee"]
    ankle_coords = rep_data["smooth_ankle"]
    wrist_smooth = rep_data["smooth_wrist"]

    ################################ ANGLE CALCULATIONS ###############################

    # Primary state-machine angle.
    # Standing (~170-180°)  →  bottom of deadlift (~60-90°).
    hip_angle = calculate_angle(shoulder_coords, hip_coords, knee_coords)

    # Knee angle for lockout check at the top.
    knee_angle = calculate_angle(hip_coords, knee_coords, ankle_coords)

    # Torso lean (back angle) — supplementary debug metric.
    back_angle = angle_to_vertical_degrees(hip_coords, shoulder_coords)

    # Torso length for bar-path normalisation.
    torso_length = np.linalg.norm(np.array(shoulder_coords) - np.array(hip_coords))

    ################################ SPINE CURVATURE (this frame) ###############################

    cur_curvature = None
    if spine_keypoints is not None and spine_scores is not None:
        spine_pts = _extract_spine_points(
            spine_keypoints, spine_scores, frame_width, frame_height
        )
        if spine_pts is not None:
            cur_curvature = compute_spine_curvature(spine_pts)

    ################################ STATES ###############################

    rep_finished = False

    if landmarks_reliable:
        # TOP POSITION
        if hip_angle > 150:
            if state == "down":
                rep_data["rep_end_time"] = video_timestamp
                rep_data["knee_angle_at_top"] = knee_angle
                rep_finished = True
            state = "up"

        # BOTTOM POSITION
        elif hip_angle < 110:
            if state == "up":
                rep_data["rep_start_time"] = video_timestamp
            state = "down"

    ################################ TRACKING (only down state) ###############################

    if state == "down" and landmarks_reliable:
        rep_data["min_hip_angle"] = min(rep_data["min_hip_angle"], hip_angle)
        rep_data["wrist_x_positions"].append(wrist_smooth[0])

        if cur_curvature is not None:
            rep_data["spine_curvatures"].append(cur_curvature)

        # mark valid only once we reach meaningful hinge depth
        if hip_angle < 90:
            rep_data["valid"] = True

    ################################ SAFE CURRENT FRAME STATS FOR DEBUG ###############################

    cur_min_hip = (
        rep_data["min_hip_angle"] if rep_data["min_hip_angle"] < 999 else hip_angle
    )
    cur_spine_curv = (
        float(np.mean(rep_data["spine_curvatures"]))
        if rep_data["spine_curvatures"]
        else (cur_curvature if cur_curvature is not None else 0.0)
    )
    cur_bar_drift = (
        (max(rep_data["wrist_x_positions"]) - min(rep_data["wrist_x_positions"]))
        / torso_length
        if rep_data["wrist_x_positions"] and torso_length > 0
        else 0.0
    )

    ################################  FEEDBACK  ###############################

    if rep_finished:
        reps += 1
        feedback_lines = []
        score = 100

        if not rep_data["valid"]:
            feedback_lines.append("Rep invalid (insufficient hip hinge depth)")
            score = 0
        else:
            # ------------------ hip hinge depth ------------------
            min_hip = rep_data["min_hip_angle"]

            if min_hip < 70:
                hinge_result = "Full depth"
            elif min_hip <= 85:
                hinge_result = "Good depth"
                score -= 10
            else:
                hinge_result = "Shallow — hinge deeper"
                score -= 30

            # ------------------ spine curvature (SpinePose) ------------------
            # Mean normalised perpendicular deviation across all reliable down-phase
            # frames.  Thresholds were derived empirically; a flat conventional
            # deadlift bottom position typically yields values < 0.05.
            if rep_data["spine_curvatures"]:
                mean_curv = float(np.mean(rep_data["spine_curvatures"]))

                if mean_curv < 0.05:
                    spine_result = "Neutral spine"
                elif mean_curv < 0.10:
                    spine_result = "Slight rounding"
                    score -= 15
                else:
                    spine_result = "Significant rounding"
                    score -= 30
            else:
                # SpinePose had insufficient confidence across the whole rep
                spine_result = "Not detected"

            # ------------------ bar path ------------------
            # Lateral wrist drift normalised by torso length.
            # Ideally the bar traces a near-vertical path close to the body.
            wx = rep_data["wrist_x_positions"]
            bar_drift = (
                (max(wx) - min(wx)) / torso_length if wx and torso_length > 0 else 0
            )

            if bar_drift < 0.10:
                bar_result = "Bar stayed close"
            elif bar_drift < 0.20:
                bar_result = "Slight bar drift"
                score -= 10
            else:
                bar_result = "Bar drifted away"
                score -= 20

            # ------------------ knee lockout ------------------
            knee_top = rep_data["knee_angle_at_top"]
            if knee_top is not None and knee_top > 160:
                lockout_result = "Locked out"
            else:
                lockout_result = "Not fully locked"
                score -= 15

            # ------------------ tempo ------------------
            rep_time = (
                rep_data["rep_end_time"] - rep_data["rep_start_time"]
                if rep_data["rep_start_time"]
                else 0
            )
            if rep_time < 0.8:
                tempo_result = "Too fast"
                score -= 12
            elif rep_time > 5.0:
                tempo_result = "Too slow"
            else:
                tempo_result = "Controlled"

            score = max(0, score)

            if score >= 90:
                grade = "Excellent"
            elif score >= 75:
                grade = "Good"
            elif score >= 55:
                grade = "Okay"
            else:
                grade = "Needs work"

            print(f"DEBUG min_hip_angle:   {min_hip:.1f}")
            if rep_data["spine_curvatures"]:
                print(f"DEBUG spine_curvature: {mean_curv:.4f}")
            else:
                print("DEBUG spine_curvature: N/A")
            print(f"DEBUG bar_drift:       {bar_drift:.3f}")
            if knee_top:
                print(f"DEBUG knee_at_top:     {knee_top:.1f}")
            else:
                print("DEBUG knee_at_top: None")
            if rep_data["rep_start_time"]:
                print(f"DEBUG rep_time:        {rep_time:.2f}s\n")

            feedback_lines.append(f"Score: {score}/100 ({grade})")
            feedback_lines.append(f"Hinge depth: {hinge_result}")
            feedback_lines.append(f"Spine: {spine_result}")
            feedback_lines.append(f"Bar path: {bar_result}")
            feedback_lines.append(f"Lockout: {lockout_result}")
            feedback_lines.append(f"Tempo: {tempo_result}")

        rep_feedback_blocks.append(
            {
                "rep": reps,
                "side": f"{side_label} side",
                "feedback": feedback_lines,
            }
        )

        # reset for next rep, preserving locked side and EMA state so smoothing
        # doesn't restart from None at the beginning of every rep
        smooth_state = {
            k: rep_data[k]
            for k in (
                "smooth_shoulder",
                "smooth_hip",
                "smooth_knee",
                "smooth_ankle",
                "smooth_wrist",
            )
        }
        rep_data = get_initial_deadlift_rep_data()
        rep_data["side"] = side_detected
        rep_data.update(smooth_state)

    ################################ RETURN ###############################

    angles = {
        "hip": (hip_angle, hip_coords),
        "knee": (knee_angle, knee_coords),
    }

    return (
        "",  # summary (unused slot, consistent with other analysers)
        hip_angle,  # primary display angle
        "Deadlift analysis",  # feedback label
        (0, 165, 255),  # orange — distinct from squat (green) and pushup (cyan)
        "",  # label (unused)
        side_label,
        angles,
        knee_angle,  # secondary angle slot
        reps,
        state,
        rep_data,
        rep_feedback_blocks,
        back_angle,  # debug
        knee_angle,  # debug
        cur_min_hip,  # debug
        cur_spine_curv,  # debug
        cur_bar_drift,  # debug
    )
