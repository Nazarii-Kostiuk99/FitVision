"""
Exercise form analysis functions
"""
from angle_calculations import calculate_angle


############################################# SQUAT ANALYSIS ################################################
def analyse_squat(landmarks, mp_pose):
    set_feedback=[]
    """
    Squat form analysis
    
    Depth categories (knee flexion):
    -> Shallow: 90-180
    -> Medium: 70-90 
    -> Deep: <70 (below parallel)
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
        side = "Right side"
    else:
        shoulder, hip, knee, ankle = l_shoulder, l_hip, l_knee, l_ankle
        side = "Left side"
        
    # utility for thigh angle calc
    if side == "Right side":
        horizontal_ref = [knee[0] + 0.3, knee[1]] #point to the right of knee for hor. ref.
    else:
        horizontal_ref = [knee[0] - 0.3, knee[1]] #point to the left of knee for hor. ref.
    
    ####################### calculation of key angles ########################
    
    hip_vertical = [hip[0], hip[1] + 0.3]  # point below hip for vert. ref.
    knee_vertical = [knee[0], knee[1] + 0.3]  # point below knee

    knee_angle = calculate_angle(hip, knee, ankle)
    thigh_angle = calculate_angle(hip, knee, horizontal_ref)
    trunk_angle = calculate_angle(shoulder, hip, hip_vertical)# trunk inclination - deviation from vertical
    shin_angle = calculate_angle(knee, ankle, knee_vertical) #shin inclination - shin angle from vertical
    trunk_shin_diff = trunk_angle - shin_angle     # trunk-shin bias calculation

    
    # determine squat bias (glutes/quads/balanced)
    if trunk_shin_diff > 10:
        bias = "Hip-biased (glutes)"
    elif trunk_shin_diff < -10:
        bias = "Knee-biased (quads)"
    else:
        bias = "Neutral bias"
    
    # check depth based on reference points (lower angle = deeper squat)
    if knee_angle < 60:
        depth = "Too deep"
        depth_colour = (0, 255, 0)
        set_feedback.append("Sqaut was too deep. Stop a bit higher to reduce stress on your knee joints. Aim for thighs to be parallel to the ground.")
    elif 60 <= knee_angle <= 110:
        depth = "Good"
        set_feedback.append("Good  depth. Thighs are about parallel to the floor.")
        depth_colour = (0, 255, 0)
    else:
        depth = "Shallow"
        set_feedback.append("Your squat is bit shallow. Try lowering your hips until thighs are parallel to the ground.")
        depth_colour = (0, 0, 255)
    
    # Thigh feedback
    if thigh_angle <= 20:
        set_feedback.append("Great range of motion. Thighs are almost parallel to ground")
    elif thigh_angle > 25:
        set_feedback.append("Try squating lower so your thighs reach a parallel line with the floor.")

    # Trunk posture
    if trunk_angle < 25:
        set_feedback.append("Great upright posture throughout squat")
    elif 25 <= trunk_angle <= 45:
        set_feedback.append("Slight forward lean while squatting. Try keeping your chest higher up.")
    else:
        set_feedback.append("Too much forward lean. Focus on keeping your chest upright facing forward and engaging your core.")

    ########################## Depth ratio #####################
    
    # standing_hip_y = max(all_hip_y_values)
    # bottom_hip_y = min(all_hip_y_values)
    # depth_ratio = (hip[1] - bottom_hip_y) / (standing_hip_y - bottom_hip_y + 1e-8)

    # if depth_ratio <= 0.55:
    #     set_feedback.append("Excellent depth control — hips drop below knee line.")
    # elif 0.55 < depth_ratio <= 0.65:
    #     set_feedback.append("Good consistent squat depth.")
    # else:
    #     set_feedback.append("Not deep enough — lower your hips a bit more for full range.")

    
    primary_feedback = f"{depth} | {bias}"
    colour = depth_colour
    display_angle = knee_angle
    label = "Knee Angle"
    summary = set_feedback
    
    return summary, display_angle, primary_feedback, colour, label, side, 

