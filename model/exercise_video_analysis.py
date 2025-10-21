import sys
import cv2
import mediapipe 
import numpy as np
from video_utility import mov_to_mp4
from angle_calculations import calculate_angle
from exercise_analysis import analyse_squat
# analyse_pushup, analyse_lunge, analyse_plank

def analyse_video(video_path, exercise_type):
    """
    Supported: squat, pushup, lunge, plank
    Exercise selected by user -> parsed to backend + called by django
    """
    
    if exercise_type not in ["squat", "pushup", "lunge", "plank"]:
        print(f"Exercise '{exercise_type}' not supported yet")
        return
    
    video_path = mov_to_mp4(video_path) # mov(iphone format) -> mp4 
    
    
    reps = 0
    state = "up"

    mp_pose = mediapipe.solutions.pose
    mp_drawing = mediapipe.solutions.drawing_utils
    
    # opencv reads a video from a file
    cap = cv2.VideoCapture(video_path)
    # if not cap.isOpened():
    #     print(f"Failed to open video file: {video_path}")
    #     return
    # else:
    #     print(f"Opened video: {video_path}")

    # record fps, dimensions of input video 
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #writer output logic
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('exercise_output.mp4', fourcc, fps, (width, height))
    
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) # test other value mb better accuracy

    printed_feedback = False  #only print once to console
    
    while cap.isOpened():
        ret, frame = cap.read()
        # print("Reading frames")

        # if end of video or no frame
        if not ret:
        #     # print("Failed to read frame or end")
         break

        #rgb for mediapipe 
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # drawing points of interest (joints)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3)
            )

            landmarks = results.pose_landmarks.landmark
            
            # call the appropriate exercise type analyser 
            if exercise_type == "squat":
                summary, angle, feedback, color, label, side, angles, depth_ratio,reps = analyse_squat(landmarks, mp_pose, reps, state="up")
                # print(summary)
            elif exercise_type == "pushup":
                angle, feedback, color, label = analyse_pushup(landmarks, mp_pose)
            elif exercise_type == "lunge":
                angle, feedback, color, label = analyse_lunge(landmarks, mp_pose)
            elif exercise_type == "plank":
                angle, feedback, color, label = analyse_plank(landmarks, mp_pose)
             
             
             
             
             
             
                
            if not printed_feedback:
                print("\n#### SET SUMMARY ####")
                for line in summary:
                    print(f"• {line}")
                printed_feedback = True

            # Display angle and feedback
            
           
            y_offset = 500 
            for i, line in enumerate(summary):
                y = y_offset + i * 40  # vert. space
                cv2.putText(frame, line, (30, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            #drawing abgles for corresponding joints
            for name, (ang, coord) in angles.items():
                x = int(coord[0] * width)
                y = int(coord[1] * height)
                cv2.putText(frame, f"{name}: {int(ang)}°",
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255), 2, cv2.LINE_AA)
                
                
            cv2.putText(frame, f"{label}: {int(angle)}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, feedback, (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"{side}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            
            
            cv2.putText(frame, f"Depth Ratio: {depth_ratio:.2f}",
            (30, 160),  # position below main feedback
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            
            
            cv2.putText(frame, f"Reps: {reps}", (30, 200),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            
            
            

        # write output video to a separate file (later send to django api -> DB)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    
#entry point   
if __name__ == "__main__":
    if len(sys.argv) > 2:
        video_file = sys.argv[1]
        exercise = sys.argv[2]
        analyse_video(video_file, exercise)
    else:
        # defaults to sqaut
        analyse_video("test_squat.mov", "squat")