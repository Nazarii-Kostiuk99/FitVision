from video_utility import mov_to_mp4
import os

input_path = "test_video.mov"  
if not os.path.exists(input_path):
    print(f"File not found: {input_path}")
    exit(1)

try:
    output_path = mov_to_mp4(input_path)
    print(f"Converted file saved as: {output_path}")
    print(f"Conversion successful: {os.path.exists(output_path)}")
except Exception as e:
    print("Error during conversion:", e)
