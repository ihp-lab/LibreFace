import cv2
from decord import VideoReader
from decord import cpu, gpu
import gdown
import os
import pandas as pd
from tqdm import tqdm

# Define a function to download the model weights
def download_weights(drive_id, model_path):
    model_dir = "/".join(model_path.split("/")[:-1])
    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(model_path):
        print(f"Downloading model weights - {model_path}...")
        gdown.download(id=drive_id, output=model_path)
        if not os.path.exists(model_path):
            print("Error occured in downloading...")
    # else:
        # print(f"{model_path} already exists. Skippind model weights download.")

    return model_path

def get_frames_from_video(video_path, temp_dir="./tmp"):
    cur_video_name = ".".join(video_path.split("/")[-1].split(".")[:-1])
    cur_video_save_path = os.path.join(temp_dir, cur_video_name)
    os.makedirs(cur_video_save_path, exist_ok=True)

    ffmpeg_command = f"ffmpeg -i {video_path} '{cur_video_save_path}/{cur_video_name}_%06d.png' -loglevel warning"
    os.system(ffmpeg_command)
    
    return cur_video_save_path

## Adapted from https://gist.github.com/HaydenFaulkner/3aa69130017d6405a8c0580c63bee8e6
def get_frames_from_video_decord(video_path, temp_dir="./temp"):

    cur_video_name = ".".join(video_path.split("/")[-1].split(".")[:-1])
    frames_dir = os.path.join(temp_dir, cur_video_name)
    os.makedirs(frames_dir, exist_ok=True)

    vr = VideoReader(video_path, ctx=cpu(0))
    start = 0
    end = len(vr)

    timestamps = vr.get_frame_timestamp(range(len(vr)))
    timestamps = (timestamps[:, 0] * 1000).round(3).astype(float).tolist()

    frame_paths = []
    frame_idxs = []

    for index in tqdm(range(start, end), desc="Reading frames from the video"):  # lets loop through the frames until the end
        frame = vr[index]  # read an image from the capture
        
        save_path = os.path.join(frames_dir, "{:010d}.png".format(index))  # create the save path
        cv2.imwrite(save_path, cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR))  # save the extracted image
        frame_paths.append(save_path)
        frame_idxs.append(index)
    
    frames_df = pd.DataFrame({"frame_idx":frame_idxs, "frame_time_in_ms":timestamps, "path_to_frame":frame_paths})
    
    return frames_df

if __name__=="__main__":
    video_path = "/home/achaubey/Desktop/projects/data/DISFA/Videos_LeftCamera/LeftVideoSN001_comp.avi"
    cur_video_frames_path, frames_df = get_frames_from_video_decord(video_path, temp_dir="./tmp")
    
    frames_df.to_csv("video_frames.csv")
    print(cur_video_frames_path)

def uniquify_dir(dir_path):
    dir_path = dir_path.rstrip("/")
    dir_name = dir_path.split("/")[-1]
    par_dir = "/".join(dir_path.split("/")[:-1])

    counter=1
    while os.path.exists(dir_path):
        dir_path = f"{par_dir}/{dir_name}_{counter}"
        counter+=1
    return dir_path

def uniquify_file(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def check_file_type(file_path):
    IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.mpeg', '.mpg']

    # Extract the file extension
    _, ext = os.path.splitext(file_path)
    
    # Convert extension to lowercase to avoid case-sensitivity issues
    ext = ext.lower()

    if ext in IMAGE_EXTENSIONS:
        return "image"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    else:
        return "unknown"