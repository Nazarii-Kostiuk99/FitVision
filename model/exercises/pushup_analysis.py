"""
Pushup form analysis (side view)

Rep counter (elbow flex angle)
Criteria:
-> Engaged core:
    -> Neutral spine
    -> Hips dont drop/pike
    -> Rigid trunk
-> Elbow flex depth
-> Shoulder stability
-> Shoulder-over-wrist
-> Neutral neck
-> Tempo
"""

from Utility.angle_calculations import calculate_angle
import statistics
import numpy as np


# stores per-rep tracking data ->  Reset after every completed rep
def get_initial_pushup_rep_data():
    return {
        "side": None,
        "min_elbow_angle": 999,
        "body_angles": [],
        "hip_positions": [],
        "shoulder_offsets": [],
        "neck_angles": [],
        "rep_start_time": None,
        "rep_end_time": None,
        "valid": False,
        "hip_y_start": None,  # hip position at start of down phase
        "min_hip_y": 999,
        "max_hip_y": 0,
    }


################################ SIDE DETECTION ###############################


def detect_side(landmarks, mp_pose):
    left_points = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_ELBOW.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value,
    ]

    right_points = [
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    ]

    left_visibility = np.mean([landmarks[i].visibility for i in left_points])

    right_visibility = np.mean([landmarks[i].visibility for i in right_points])

    if left_visibility >= right_visibility:
        return "left"
    else:
        return "right"


def analyse_pushup(landmarks, mp_pose, reps, state, rep_data, rep_feedback_blocks, video_timestamp=0.0):

    # settings side after detection
    if rep_data["side"] is None:
        rep_data["side"] = detect_side(landmarks, mp_pose)

    side_detected = rep_data["side"]

    if side_detected == "left":
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        side_label = "Left"
    else:
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
        side_label = "Right"

    shoulder_coords = [shoulder.x, shoulder.y]
    elbow_coords = [elbow.x, elbow.y]
    wrist_coords = [wrist.x, wrist.y]
    hip_coords = [hip.x, hip.y]
    ankle_coords = [ankle.x, ankle.y]
    ear_coords = [ear.x, ear.y]

    ################################ ANGLE CALCULATIONS ###############################

    elbow_angle = calculate_angle(shoulder_coords, elbow_coords, wrist_coords)
    body_angle = calculate_angle(shoulder_coords, hip_coords, ankle_coords)
    neck_angle = calculate_angle(ear_coords, shoulder_coords, hip_coords)

    ######## SHOULDER OVER WRIST (normalised) ##########

    torso_length = np.linalg.norm(np.array(shoulder_coords) - np.array(hip_coords))

    shoulder_offset = abs(shoulder.x - wrist.x)
    shoulder_offset_norm = shoulder_offset / torso_length if torso_length > 0 else 0

    ################################ REP DATA ###############################

    # rep_data["min_elbow_angle"] = min(rep_data["min_elbow_angle"], elbow_angle)
    # rep_data["body_angles"].append(body_angle)
    # rep_data["hip_positions"].append(hip.y)
    # rep_data["shoulder_offsets"].append(shoulder_offset_norm)

    ################################ STATES ###############################

    rep_finished = False

    # TOP POSITION
    if elbow_angle > 160:
        if state == "down":
            rep_data["rep_end_time"] = video_timestamp
            rep_finished = True

        state = "up"

    # BOTTOM POSITION
    elif elbow_angle < 140:
        if state == "up":
            rep_data["rep_start_time"] = video_timestamp
            rep_data["hip_y_start"] = hip.y
        state = "down"

    ################################ TRACKING (only while down) ###############################

    if state == "down":
        # if rep_data["hip_y_start"] is None:
        #     rep_data["hip_y_start"] = hip.y
        rep_data["min_hip_y"] = min(rep_data["min_hip_y"], hip.y)
        rep_data["max_hip_y"] = max(rep_data["max_hip_y"], hip.y)
        rep_data["min_elbow_angle"] = min(rep_data["min_elbow_angle"], elbow_angle)
        rep_data["body_angles"].append(body_angle)
        rep_data["hip_positions"].append(hip.y)
        rep_data["shoulder_offsets"].append(shoulder_offset_norm)
        rep_data["neck_angles"].append(neck_angle)

        # mark valid only once we reach meaningful depth
        if elbow_angle < 120:
            rep_data["valid"] = True

    ################################ SAFE CURRENT FRAME STATS FOR DEBUG ###############################

    cur_min_body = (
        min(rep_data["body_angles"]) if rep_data["body_angles"] else body_angle
    )
    cur_max_body = (
        max(rep_data["body_angles"]) if rep_data["body_angles"] else body_angle
    )

    ################################ BUILD FEEDBACK WHEN REP FINISHES ###############################

    if rep_finished:
        reps += 1
        feedback_lines = []
        score = 100

        if not rep_data["valid"]:
            feedback_lines.append("Rep invalid (insufficient depth)")
            score = 0
        else:
            # ------------------ elbow flex (depth) ------------------
            min_elbow = rep_data["min_elbow_angle"]

            if min_elbow < 80:
                depth_result = "Excellent depth"
            elif min_elbow <= 95:
                depth_result = "Good depth"
                score -= 15
            else:
                depth_result = "Shallow"
                score -= 45

            # |------------------ core (hip sag/pike) ------------------|

            # LANDMARKS: Shoulder, Hip, Ankle.
            # ANGLE: If the Shoulder-Hip-Ankle angle < 158°, the form is breaking.
            #    - Low h Y is (small number), the hip is high in the air (PIKE).
            #    - If Hip Y is HIGH (large number), the hip is dropping toward the floor (SAG).
            avg_hip_y = statistics.mean(rep_data["hip_positions"])

            hip_drop = rep_data["max_hip_y"] - rep_data["hip_y_start"]  # positive = sag
            hip_rise = (
                rep_data["hip_y_start"] - rep_data["min_hip_y"]
            )  # positive = pike

            min_body = cur_min_body
            max_body = cur_max_body

            if min_body < 145:
                if avg_hip_y < 0.36:
                    core_result = "Hips piked"
                    score -= 30
                else:
                    core_result = "Hips sagging"
                    score -= 30
            elif min_body < 158:
                if avg_hip_y < 0.38:
                    core_result = "Slight hip pike"
                    score -= 15
                else:
                    core_result = "Slight hip sag"
                    score -= 15
            elif max_body > 195:
                # extreme pike
                core_result = "Hips piked"
                score -= 30
            else:
                core_result = "Neutral spine"

            # ------------------ shoulders ------------------
            max_offset = max(rep_data["shoulder_offsets"])

            if max_offset < 0.25:
                shoulder_result = "Good alignment"
            elif max_offset < 0.35:
                shoulder_result = "Slightly forward"
                score -= 8
            else:
                shoulder_result = "Too far forward"
                score -= 20

            # ------------------ neck ------------------
            avg_neck = float(np.mean(rep_data["neck_angles"])) if rep_data["neck_angles"] else neck_angle
            if 115 <= avg_neck <= 155:
                neck_result = "Neutral"
            else:
                neck_result = "Misaligned"
                score -= 8

            # ------------------ tempo ------------------
            rep_time = (
                rep_data["rep_end_time"] - rep_data["rep_start_time"]
                if rep_data["rep_start_time"]
                else 0
            )

            if rep_time < 0.8:
                tempo_result = "Too fast"
                score -= 12
            elif rep_time > 4.0:
                tempo_result = "Too slow"
            else:
                tempo_result = "Controlled"

            score = max(0, score)

            if score >= 90:
                grade = "Excellent"
            elif score >= 78:
                grade = "Good"
            elif score >= 60:
                grade = "Okay"
            else:
                grade = "Needs work"

            print(f"DEBUG min_elbow: {min_elbow:.1f}")
            print(f"DEBUG min_body: {cur_min_body:.1f}")
            print(f"DEBUG max_body: {cur_max_body:.1f}")
            print(f"DEBUG max_shoulder_offset: {max(rep_data['shoulder_offsets']):.3f}")
            # print(f"DEBUG hip_rise: {hip_rise:.1f}")
            # print(f"DEBUG hip_drop: {hip_drop:.1f}")
            # print(f"DEBUG hip_y: {hip.y:.3f}  body_angle: {body_angle:.1f}")
            print(f"DEBUG avg_hip_y: {avg_hip_y:.3f}")
            print(f"DEBUG max_shoulder_offset: {max(rep_data['shoulder_offsets']):.3f}")
            if rep_data["rep_start_time"]:
                rep_time = rep_data["rep_end_time"] - rep_data["rep_start_time"]
                print(f"DEBUG rep_time: {rep_time:.2f}s\n")

            feedback_lines.append(f"Score: {score}/100 ({grade})")
            feedback_lines.append(f"Depth: {depth_result}")
            feedback_lines.append(f"Core: {core_result}")
            feedback_lines.append(f"Shoulder: {shoulder_result}")
            feedback_lines.append(f"Neck: {neck_result}")
            feedback_lines.append(f"Tempo: {tempo_result}")

        rep_feedback_blocks.append(
            {
                "rep": reps,
                "side": f"{side_label} side",
                "feedback": feedback_lines,
            }
        )

        # reset for next rep, keep side so it doesnt re-detect every rep
        rep_data = get_initial_pushup_rep_data()
        rep_data["side"] = side_detected

    ################################ RETURN ###############################

    angles = {
        "elbow": (elbow_angle, elbow_coords),
        "body": (body_angle, hip_coords),
    }

    return (
        elbow_angle,
        "Pushup analysis",
        (0, 255, 255),
        side_label,
        angles,
        rep_data["min_elbow_angle"],
        reps,
        state,
        rep_data,
        rep_feedback_blocks,
        shoulder_offset_norm,
        body_angle,
        neck_angle,
        cur_min_body,
        cur_max_body,
    )
