import cv2
# from decord import VideoReader  ## decord causes issues with torch running on GPU
# from decord import cpu          ## Uncomment these lines only when you really want to
import gdown
import os
import pandas as pd
import subprocess
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

def get_frames_from_video_ffmpeg(video_path, temp_dir="./tmp"):
    
    cur_video_name = ".".join(video_path.split("/")[-1].split(".")[:-1])
    cur_video_save_path = uniquify_dir(os.path.join(temp_dir, cur_video_name))
    os.makedirs(cur_video_save_path, exist_ok=True)

    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # FFmpeg command to extract frames
    output_pattern = os.path.join(cur_video_save_path, 'frame_%09d.png')
    ffmpeg_command = [
        'ffmpeg', '-i', video_path,
        output_pattern
    ]
    
    # Run the ffmpeg command
    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Create DataFrame to store frame index, path, and timestamp
    frame_files = sorted(os.listdir(cur_video_save_path))
    frame_index = []
    frame_paths = []
    frame_timestamps = []

    # Get frame rate of the video
    ffprobe_command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 
        'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', video_path
    ]
    fps_output = subprocess.check_output(ffprobe_command, text=True).strip()
    fps = eval(fps_output)  # Get frames per second as a float

    # Populate DataFrame
    for i, frame_file in enumerate(frame_files):
        frame_index.append(i)
        frame_paths.append(os.path.join(cur_video_save_path, frame_file))
        frame_timestamps.append(i / fps)

    # Create DataFrame
    df = pd.DataFrame({
        'frame_idx': frame_index,
        'frame_time_in_ms': frame_timestamps,
        'path_to_frame': frame_paths
    })
    
    return df

def get_frames_from_video_opencv(video_path, temp_dir="./tmp"):
    cur_video_name = ".".join(video_path.split("/")[-1].split(".")[:-1])
    cur_video_save_path = os.path.join(temp_dir, cur_video_name)
    os.makedirs(cur_video_save_path, exist_ok=True)

    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()

    count=0
    frame_paths = []
    frame_idxs = []
    frame_timestamps = []

    while(success):
        cur_frame_path = os.path.join(cur_video_save_path, "{:010d}.png".format(count))
        frame_timestamps.append(round(vidcap.get(cv2.CAP_PROP_POS_MSEC), 3))
        frame_paths.append(cur_frame_path)
        frame_idxs.append(count)
        cv2.imwrite(cur_frame_path, image)
        success, image = vidcap.read()
        count+=1
    
    frames_df = pd.DataFrame({"frame_idx":frame_idxs, "frame_time_in_ms":frame_timestamps, "path_to_frame":frame_paths})
    
    return frames_df

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
    
def restructure_landmark_dict(lm_dict):
    new_lm_dict = {}
    for k, v in lm_dict.items():
        for lm_idx, lm_v in enumerate(v):
            new_lm_dict[f"{k}_{lm_idx}"] = lm_v
    return new_lm_dict

def restructure_landmark_mediapipe(lm_object):
    lm_dict = {}
    for idx, lmi in enumerate(lm_object):
        lm_dict[f"lm_mp_{idx}_x"] = lmi.x
        lm_dict[f"lm_mp_{idx}_y"] = lmi.y
        lm_dict[f"lm_mp_{idx}_z"] = lmi.z
    return lm_dict


if __name__=="__main__":
    video_path = "/home/achaubey/Desktop/projects/data/DISFA/Videos_LeftCamera/LeftVideoSN001_comp.avi"
    cur_video_frames_path, frames_df = get_frames_from_video_decord(video_path, temp_dir="./tmp")
    
    frames_df.to_csv("video_frames.csv")
    print(cur_video_frames_path)