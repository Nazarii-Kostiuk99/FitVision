import ffmpeg
import os

"""
    Utiity function to convert a video .mov to .mp4.
    If already .mp4, returns input_path.
"""
def mov_to_mp4(input_video_path):
    if input_video_path.lower().endswith('.mp4'):
        return input_video_path

    base, _ = os.path.splitext(input_video_path)
    output_path = f"{base}.mp4"

    try:
        (
            ffmpeg
            .input(input_video_path)
            .output(output_path, vcodec='libx264', acodec='aac', strict='experimental')
            .run(overwrite_output=True, quiet=True)
        )
        return output_path
    except ffmpeg.Error as e:
        print("Error converting video:", e)
        raise
    
    
    