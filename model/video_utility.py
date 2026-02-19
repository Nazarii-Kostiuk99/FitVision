import ffmpeg
import os
import cv2
import mediapipe

"""
    Utiity function to convert a video .mov to .mp4.
    If alrdy .mp4 -> input_path.
"""


def mov_to_mp4(input_video_path):
    if input_video_path.lower().endswith(".mp4"):
        return input_video_path

    base, _ = os.path.splitext(input_video_path)
    output_path = f"{base}.mp4"

    try:
        (
            ffmpeg.input(input_video_path)
            .output(output_path, vcodec="libx264", acodec="aac", strict="experimental")
            .run(overwrite_output=True, quiet=True)
        )
        return output_path
    except ffmpeg.Error as e:
        print("Error converting video:", e)
        raise


# face blurring after conversion
def anonymise_video(input_video_path):
    blur_margin = 0.5
    base, ext = os.path.splitext(input_video_path)
    output_path = f"{base}_anonymised{ext}"

    # for consistent blur box when lost track of face frames
    last_bbox = None
    no_face_frames = 0
    allowed_no_face_frames = 10

    mp_face = mediapipe.solutions.face_detection
    detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open {input_video_path}")

    # prep writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # process till end of video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        current_bbox = None

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            x1 = int((bbox.xmin - blur_margin * bbox.width) * width)
            y1 = int((bbox.ymin - blur_margin * bbox.height) * height)
            x2 = int((bbox.xmin + (1 + blur_margin) * bbox.width) * width)
            y2 = int((bbox.ymin + (1 + blur_margin) * bbox.height) * height)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            current_bbox = (x1, y1, x2, y2)
            last_bbox = current_bbox
            no_face_frames = 0
        else:
            no_face_frames += 1

        # use last calculated box if no face frames for 5 frames (avoids flcikering of bbox)
        if last_bbox and no_face_frames < allowed_no_face_frames:
            x1, y1, x2, y2 = last_bbox
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(face, (71, 71), 0)

        out.write(frame)

    cap.release()
    out.release()
    detector.close()
    return output_path


def preprocess_video(video_input_path):
    mp4_path = mov_to_mp4(video_input_path)
    anonymised_path = anonymise_video(mp4_path)
    return anonymised_path
