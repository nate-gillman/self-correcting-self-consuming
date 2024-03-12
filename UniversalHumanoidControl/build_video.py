import os
import sys
import cv2
from collections import defaultdict

def build_video(dir_path, frame_rate):
    # Get all PNG files in the directory
    all_file_names = [f for f in os.listdir(dir_path) if f.endswith('.png')]


    envs_to_files = defaultdict(list)
    for file_name in all_file_names:

        env_identifier = file_name[:8]
        envs_to_files[env_identifier].append(file_name)

    for env_identifier in envs_to_files:

        file_names = envs_to_files[env_identifier]

        # Sort the file names by their numerical order
        file_names.sort(key=lambda x: str(x.split('.')[0]))

        # Get the first image to determine the video dimensions
        img = cv2.imread(os.path.join(dir_path, file_names[0]))
        height, width, _ = img.shape

        # Initialize the video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(dir_path, f'-output-{env_identifier}.mp4')
        video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))

        # Write each frame to the video
        for file_name in file_names:
            img = cv2.imread(os.path.join(dir_path, file_name))
            video_writer.write(img)

        print(f"Wrote to {video_path}")

        # Release the video writer
        video_writer.release()

if __name__ == '__main__':
    # Get the directory path and frame rate from command line arguments
    dir_path = sys.argv[1]
    frame_rate = int(sys.argv[2])

    # Build the video
    build_video(dir_path, frame_rate)