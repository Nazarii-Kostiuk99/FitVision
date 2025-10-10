"""
Exercise form analysis functions
"""

from angle_calculations import calculate_angle

def analyse_squat(landmarks, mp_pose):
    """
    Squat form analysis
    
    Depth categories (knee flexion):
    -> Shallow: 90-180
    -> Medium: 70-90 
    -> Deep: <70 (below parallel)
    
    Research shows trunk-tibia angle difference indicates loading:
    - Trunk > Tibia by >10° = hip-biased (glutes/hamstrings)
    - Tibia > Trunk by >10° = knee-biased (quads)
    - Within ±10 = neutral bias
    """
    
    #landmarks for squat (both l/r so both can be analysed)
    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
             landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
              landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    
    # check visibility (to determine side l/r)
    r_visibility = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
    l_visibility = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility
    
    # choose side with better visibility
    if r_visibility > l_visibility:
        shoulder, hip, knee, ankle = r_shoulder, r_hip, r_knee, r_ankle
    else:
        shoulder, hip, knee, ankle = l_shoulder, l_hip, l_knee, l_ankle
    
    # calculate key angles
    knee_angle = calculate_angle(hip, knee, ankle)
    
    # trunk inclination - angle from vertical
    hip_vertical = [hip[0], hip[1] + 0.3]  # point below hip for vertical ref.
    trunk_angle = calculate_angle(shoulder, hip, hip_vertical)
    
    # shin inclination - shin angle from vertical
    knee_vertical = [knee[0], knee[1] + 0.3]  # point below knee
    tibia_angle = calculate_angle(knee, ankle, knee_vertical)
    
    # trunk-shin bias calculation
    trunk_tibia_diff = trunk_angle - tibia_angle
    
    # determine squat bias (glutes/quads/balanced)
    if trunk_tibia_diff > 10:
        bias = "Hip-biased (glutes)"
        depth_colour = (255, 165, 0)  # orange
    elif trunk_tibia_diff < -10:
        bias = "Knee-biased (quads)"
        depth_colour = (147, 112, 219)  # purple
    else:
        bias = "Neutral bias"
        depth_colour = (0, 255, 255)  # cyan
    
    # check depth based on reference points (lower angle = deeper squat)
    if knee_angle < 70:
        depth = "Deep squat"
        depth_colour = (0, 255, 0)
    elif knee_angle < 90:
        depth = "Good depth (parallel)"
        depth_colour = (0, 255, 0)
    elif knee_angle < 110:
        depth = "Shallow - go deeper"
        depth_colour = (255, 255, 0)  # yellow
    else:
        depth = "Too shallow"
        depth_colour = (0, 0, 255)
    
    # trunk lean check (excessive forward lean can stress lower back)
    if trunk_angle > 50:
        trunk_feedback = "Too much forward lean"
        feedback_colour = (0, 0, 255)
    elif trunk_angle > 35:
        trunk_feedback = "Moderate lean - OK"
        feedback_colour = (255, 165, 0)
    else:
        trunk_feedback = "Good upright posture"
        feedback_colour = (0, 255, 0)
    
    # primary feedback priority: safety first, then depth
    if trunk_angle > 50:
        primary_feedback = trunk_feedback
        colour = feedback_colour
        display_angle = trunk_angle
        label = "Trunk Angle"
    else:
        primary_feedback = f"{depth} | {bias}"
        colour = depth_colour
        display_angle = knee_angle
        label = "Knee Angle"
    
    return display_angle, primary_feedback, colour, label


# def analyse_pushup(landmarks, mp_pose):
#     """
#     Pushup form checker
#     Tracks elbow angle and body alignment
#     """
#     shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#     elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
#              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
#     wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
#              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    
#     elbow_angle = calculate_angle(shoulder, elbow, wrist)
    
#     # also check body alignment
#     hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
#            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
#     ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
#              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
#     body_angle = calculate_angle(shoulder, hip, ankle)
    
#     # prioritize body alignment feedback
#     if body_angle < 160:
#         feedback = "Keep back straight!"
#         colour = (255, 0, 0)
#     elif elbow_angle < 90:
#         feedback = "Good depth"
#         colour = (0, 255, 0)
#     else:
#         feedback = "Go lower"
#         colour = (0, 0, 255)
    
#     return elbow_angle, feedback, colour, "Elbow Angle"


# def analyse_lunge(landmarks, mp_pose):
#     """
#     Lunge form checker
#     Front knee should hit ~90 degrees, shouldn't go past toes
#     """
#     # front leg (right)
#     hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
#            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
#     knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
#             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
#     ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
#              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
#     angle = calculate_angle(hip, knee, ankle)
    
#     if angle < 85 or angle > 95:
#         feedback = "Aim for 90° angle"
#         colour = (255, 165, 0)  # orange
#     else:
#         feedback = "Perfect form!"
#         colour = (0, 255, 0)
    
#     return angle, feedback, colour, "Front Knee"


# def analyse_plank(landmarks, mp_pose):
#     """
#     Plank form checker
#     Body should form straight line - no sagging hips
#     """
#     shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#     hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
#            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
#     ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
#              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    
#     body_angle = calculate_angle(shoulder, hip, ankle)
    
#     # closer to 180 = straighter
#     if body_angle > 170:
#         feedback = "Perfect alignment!"
#         colour = (0, 255, 0)
#     elif body_angle > 160:
#         feedback = "Good form"
#         colour = (0, 255, 0)
#     else:
#         feedback = "Straighten body"
#         colour = (0, 0, 255)
    
#     return body_angle, feedback, colour, "Body Line"