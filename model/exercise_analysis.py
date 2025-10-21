"""
Exercise form analysis functions
"""
from angle_calculations import calculate_angle



############################################# SQUAT ANALYSIS ################################################
def analyse_squat(landmarks, mp_pose, reps, state):
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
    thigh_angle = calculate_angle(hip, knee, horizontal_ref)#thigh vs imaginary hor.line (how tilted relative to ground)
    trunk_angle = calculate_angle(shoulder, hip, hip_vertical)# trunk inclination - deviation from vertical
    shin_angle = calculate_angle(knee, ankle, knee_vertical) #shin inclination - shin angle from vertical
    trunk_shin_diff = trunk_angle - shin_angle     # trunk-shin bias calculation
    verticality = abs(180 - trunk_angle)
    hip_y = hip[1] #vert. pos. of hip
    knee_y = knee[1] #vert. pos. of knee
    shoulder_y = shoulder[1] #vert. pos. of shoudler
    
    
    ####################### squat analysis ########################
    
    #depth
    depth_ratio = (shoulder_y - hip_y) / (shoulder_y - knee_y + 1e-6)
    if depth_ratio > 1.2: #play around with this value for most accurate results
        depth = "Too deep" #hips below knees
        depth_colour = (0, 255, 0)
        set_feedback.append("Sqaut was too deep. Stop a bit higher to reduce stress on your knee joints. Aim for thighs to be parallel to the ground.")
        
    elif 0.7 <= depth_ratio <= 1.2: #hips approx. at same level with knees
        depth = "Good"
        depth_colour = (0, 255, 0)
        set_feedback.append("Good  depth. Keep it up.")
        
    elif 0.5 <= depth_ratio < 0.7: #hips above knees
        depth = "Shallow"
        depth_colour = (0, 0, 255)
        set_feedback.append("Your squat is bit shallow. Try lowering your hips until thighs are parallel to the ground.")
    
    else:
        depth = "Not detected"
        depth_colour = (255, 255, 255)
        set_feedback.append("No squat detected yet — start your set to begin analysis.")

    # Thigh feedback
    if thigh_angle >= 160:
        set_feedback.append("Great range of motion. Thighs are parallel to ground")
    elif thigh_angle >= 100:
        set_feedback.append("Try squating lower so your thighs reach a parallel line with the floor.")

    # Trunk posture
    if verticality < 15:
        set_feedback.append("Great upright posture throughout squat")
    elif 15 <= verticality < 35:
        set_feedback.append("Slight forward lean while squatting. Try keeping your chest higher up.")
    else:
        set_feedback.append("Too much forward lean. Focus on keeping your chest upright facing forward and engaging your core.")

  ####################### repetitions counter ########################
  
    #resets the state if incorrectly set to down
    if state == "up" and depth_ratio > 0.5 and knee_angle < 100:
        state = "down"

    # Coming back up — counts as one rep
    elif state == "down" and depth_ratio < 0.6 and knee_angle > 160:
        state = "up"
        reps += 1
  
  
  
  
  
    
    #unpack with these in analyse()
    primary_feedback = f"{depth}"
    colour = depth_colour
    display_angle = knee_angle
    label = "Knee Angle"
    summary = set_feedback
    
    angles = {
    "knee": (knee_angle, knee),
    "thigh": (thigh_angle, [(hip[0] + knee[0]) / 2, (hip[1] + knee[1]) / 2]),
    "trunk": (trunk_angle, hip),
    "shin": (shin_angle, [(knee[0] + ankle[0]) / 2, (knee[1] + ankle[1]) / 2])
}
    
    return summary, display_angle, primary_feedback, colour, label, side, angles, depth_ratio, reps, state

