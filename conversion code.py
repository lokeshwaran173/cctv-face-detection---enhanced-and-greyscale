import os
import subprocess

input_file = "20th oct feed/0700-0800/073123-00074-M-0012.TS"  # Replace with the path to your TS file
output_file = "desired_filename.mp4"  # Replace with the desired output file name

ffmpeg_command = [
    "ffmpeg",
    "-i", input_file,
    "-c:v", "libx264",
    "-c:a", "aac",
    "-strict", "experimental",
    output_file
]

subprocess.call(ffmpeg_command)

print("Conversion complete.")
