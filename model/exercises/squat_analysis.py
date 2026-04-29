"""
Exercise form analysis functions
"""

from Utility.angle_calculations import (
    calculate_angle,
    angle_to_vertical_degrees,
)
import statistics


############################################# SQUAT ANALYSIS ################################################


def detect_side(landmarks, mp_pose):
    left_points = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
    ]

    right_points = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]

    left_visibility = sum(landmarks[i].visibility for i in left_points) / len(
        left_points
    )
    right_visibility = sum(landmarks[i].visibility for i in right_points) / len(
        right_points
    )

    if left_visibility >= right_visibility:
        return "left"
    else:
        return "right"


# stores per-rep tracking data ->  Reset after every completed rep
def get_initial_squat_rep_data():
    return {
        "max_depth_ratio": 0,
        "min_knee_angle": 999,
        "max_knee_angle": 0,  # peak knee angle (standing position) for ROM calculation
        "lean_angles": [],
        "max_thigh_angle": 0,
        "lowest_point_angles": None,  # store angles dict at lowest point
        "side": None,
        "heels_lifted": False,
        "valid": False,
    }


def rep_score(depth_result, thigh_result, posture_result, heels_result, knee_rom=None):
    score = 100

    # Depth
    if depth_result == "Shallow":
        score -= 25
    elif depth_result == "Too deep":
        score -= 10
    elif depth_result == "Not clearly detected":
        score -= 20

    # Thighs
    if thigh_result == "Not reaching parallel":
        score -= 20
    elif thigh_result == "Thighs are slightly above parallel":
        score -= 10

    # Posture
    if posture_result == "Slight forward lean":
        score -= 10
    elif posture_result == "Too much forward lean":
        score -= 25

    # Heels
    if heels_result.startswith("Heels lifted"):
        score -= 15

    # Knee ROM
    if knee_rom is not None:
        if knee_rom < 60:
            score -= 20
        elif knee_rom < 80:
            score -= 10

    score = max(0, min(100, score))

    # Grade
    if score >= 90:
        grade = "Excellent"
    elif score >= 75:
        grade = "Good"
    elif score >= 55:
        grade = "Okay"
    else:
        grade = "Needs work"

    return score, grade


def analyse_squat(landmarks, mp_pose, reps, state, rep_data, rep_feedback_blocks):
    # set_feedback = []

    """ """

    # landmarks for squat (both l/r so both can be analysed)
    r_shoulder = [
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
    ]
    r_hip = [
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
    ]
    r_knee = [
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
    ]
    r_ankle = [
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
    ]

    l_shoulder = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
    ]
    l_hip = [
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
    ]
    l_knee = [
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
    ]
    l_ankle = [
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
    ]

    r_heel = [
        landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y,
    ]
    r_foot_index = [
        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
        landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y,
    ]

    l_heel = [
        landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y,
    ]
    l_foot_index = [
        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y,
    ]

    # lock side on first frame
    if rep_data["side"] is None:
        rep_data["side"] = detect_side(landmarks, mp_pose)

    side_detected = rep_data["side"]

    if side_detected == "right":
        shoulder, hip, knee, ankle, heel, foot_index = (
            r_shoulder,
            r_hip,
            r_knee,
            r_ankle,
            r_heel,
            r_foot_index,
        )
        side = "Right side"
    else:
        shoulder, hip, knee, ankle, heel, foot_index = (
            l_shoulder,
            l_hip,
            l_knee,
            l_ankle,
            l_heel,
            l_foot_index,
        )
        side = "Left side"

    ####################### calculation of key angles ########################

    knee_angle = calculate_angle(hip, knee, ankle)
    thigh_angle = angle_to_vertical_degrees(knee, hip)
    hip_y = hip[1]  # vert. pos. of hip
    knee_y = knee[1]  # vert. pos. of knee
    shoulder_y = shoulder[1]  # vert. pos. of shoudler

    depth_ratio = (shoulder_y - hip_y) / (shoulder_y - knee_y + 1e-6)

    # heels
    heel_lift_threshold = 0.03
    heel_lifted_current = (foot_index[1] - heel[1]) > heel_lift_threshold

    # lean
    lean_angle = angle_to_vertical_degrees(hip, shoulder)

    ####################### squat analysis ########################
    angles = {
        "knee": (knee_angle, knee),
        "thigh": (thigh_angle, [(hip[0] + knee[0]) / 2, (hip[1] + knee[1]) / 2]),
    }

    ####################### REP STATE MACHINE & TRACKING ########################
    if state == "down":
        rep_data["valid"] = True
        rep_data["max_depth_ratio"] = max(rep_data["max_depth_ratio"], depth_ratio)
        rep_data["min_knee_angle"] = min(rep_data["min_knee_angle"], knee_angle)
        rep_data["max_thigh_angle"] = max(rep_data["max_thigh_angle"], thigh_angle)
        rep_data["lean_angles"].append(lean_angle)
        rep_data["heels_lifted"] = rep_data["heels_lifted"] or heel_lifted_current

    # state transitions
    UP_THRESHOLD = 140
    DOWN_THRESHOLD = 110

    # track peak angle every frame — the max across the full rep is the standing angle
    rep_data["max_knee_angle"] = max(rep_data["max_knee_angle"], knee_angle)

    rep_finished = False
    if state == "up" and knee_angle < DOWN_THRESHOLD:
        state = "down"

    elif state == "down" and knee_angle > UP_THRESHOLD:
        state = "up"
        reps += 1
        rep_finished = True
    # if state == "up" and depth_ratio > 0.5 and knee_angle < 100:
    #     state = "down"
    # elif state == "down" and depth_ratio < 0.6 and knee_angle > 160:
    #     state = "up"
    #     reps += 1
    #     rep_finished = True

    ####################### BUILD FEEDBACK ONLY WHEN REP FINISHES ########################

    if rep_finished and rep_data["valid"]:
        # print(f"DEBUG max_thigh: {rep_data['max_thigh_angle']:.1f}")
        rep_feedback = []

        max_depth = rep_data["max_depth_ratio"]
        # min_thigh = rep_data["min_thigh_angle"]
        # worst_verticality = rep_data["max_verticality"]

        heels_lifted = rep_data["heels_lifted"]

        # ---------------- DEPTH  ----------------
        if max_depth > 1.2:
            depth_result = "Too deep"
        elif 0.7 <= max_depth <= 1.2:
            depth_result = "Good depth"
        elif 0.6 <= max_depth < 0.7:
            depth_result = "Shallow"
        else:
            depth_result = "Not clearly detected"

        # ---------------- THIGHS  ----------------
        max_thigh = rep_data["max_thigh_angle"]
        if max_thigh >= 90:
            thigh_result = "Thighs are below parallel"
        elif max_thigh >= 75:
            thigh_result = "Great range of motion. Thighs are parallel to the ground"
        elif max_thigh >= 55:
            thigh_result = "Thighs are slightly above parallel"
        else:
            thigh_result = "Not reaching parallel"

        # ---------------- POSTURE  ----------------
        if len(rep_data["lean_angles"]) > 0:
            rep_lean = statistics.median(rep_data["lean_angles"])
        else:
            rep_lean = 0

        if rep_lean < 35:
            posture_result = "Great upright posture"
        elif rep_lean < 50:
            posture_result = "Slight forward lean"
        elif rep_lean < 65:
            posture_result = "Forward lean"
        else:
            posture_result = "Too much forward lean"

        # ---------------- HEELS  ----------------
        if heels_lifted:
            heels_result = "Heels lifted — keep weight mid-foot / heel"
        else:
            heels_result = "Stable"
        # knee ROM = peak standing angle minus deepest angle reached
        knee_rom = rep_data["max_knee_angle"] - rep_data["min_knee_angle"]

        score, grade = rep_score(
            depth_result, thigh_result, posture_result, heels_result, knee_rom
        )

        rep_feedback.append(f"Score: {score}/100 ({grade})")
        rep_feedback.append(f"Depth: {depth_result}")
        rep_feedback.append(f"Thighs: {thigh_result}")
        rep_feedback.append(f"Posture: {posture_result}")
        rep_feedback.append(f"Heels: {heels_result}")
        rep_feedback.append(f"Knee ROM: {int(knee_rom)} deg")

        rep_feedback_blocks.append(
            {
                "score": score,
                "grade": grade,
                "rep": reps,
                "side": side,
                "feedback": rep_feedback,
                "max_depth_ratio": rep_data["max_depth_ratio"],
                "min_knee_angle": rep_data["min_knee_angle"],
                "max_knee_angle": rep_data["max_knee_angle"],
                "knee_rom": knee_rom,
                "lean_median": rep_lean,
                "max_thigh_angle": rep_data["max_thigh_angle"],
                "depth_result": depth_result,
                "thigh_result": thigh_result,
                "posture_result": posture_result,
                "heels_result": heels_result,
            }
        )

        # reset for next rep, keep side locked
        rep_data = get_initial_squat_rep_data()
        rep_data["side"] = side_detected

    ####################### PER-FRAME DISPLAY FEEDBACK ########################

    if state == "down":
        worst_lean = (
            max(rep_data["lean_angles"]) if rep_data["lean_angles"] else lean_angle
        )
        if worst_lean >= 65:
            primary_feedback = "Too much forward lean"
            colour = (60, 60, 220)
        elif worst_lean >= 50:
            primary_feedback = "Forward lean"
            colour = (60, 60, 220)
        elif worst_lean >= 35:
            primary_feedback = "Slight forward lean"
            colour = (0, 165, 255)
        elif heel_lifted_current:
            primary_feedback = "Heels lifting!"
            colour = (0, 165, 255)
        elif depth_ratio < 0.6 and knee_angle < 120:
            primary_feedback = "Go deeper"
            colour = (0, 165, 255)
        else:
            primary_feedback = "Good form"
            colour = (80, 200, 80)
    else:
        primary_feedback = "Ready"
        colour = (255, 255, 255)

    return (
        knee_angle,
        primary_feedback,
        colour,
        side,
        angles,
        depth_ratio,
        reps,
        state,
        rep_data,
        rep_feedback_blocks,
        heel_lifted_current,
        lean_angle,
    )
