import pandas as pd
from tqdm import tqdm

from libreface.detect_mediapipe_image import *
from libreface.AU_Detection.inference import detect_action_units
from libreface.AU_Recognition.inference import get_au_intensities
from libreface.Facial_Expression_Recognition.inference import get_facial_expression
from libreface.utils import get_frames_from_video_decord, uniquify_file, check_file_type

def get_facial_attributes_image(image_path, temp_dir="./tmp", device="cpu"):
    print(f"Using device: {device} for inference...")
    aligned_image_path = get_aligned_image(image_path, temp_dir=temp_dir, verbose=True)
    detected_aus = detect_action_units(aligned_image_path, device = device)
    au_intensities = get_au_intensities(aligned_image_path, device = device)
    facial_expression = get_facial_expression(aligned_image_path, device = device)
    return {"input_image_path": image_path,
            "aligned_image_path": aligned_image_path,
            "detected_aus": detected_aus,
            "au_intensities": au_intensities,
            "facial_expression": facial_expression}

def get_facial_attributes_video(video_path, 
                                temp_dir="./tmp", 
                                device="cpu"):
    print(f"Using device: {device} for inference...")
    frames_df = get_frames_from_video_decord(video_path, temp_dir=temp_dir)
    cur_video_name = ".".join(video_path.split("/")[-1].split(".")[:-1])
    aligned_frames_path_list = get_aligned_video_frames(frames_df, temp_dir=os.path.join(temp_dir, cur_video_name))
    frames_df["aligned_frame_path"] = aligned_frames_path_list

    detected_aus, au_intensities, facial_expression = [], [], []
    for _, row in tqdm(frames_df.iterrows(), "Detecting action units and facial expression on video frames..."):
        detected_aus.append(detect_action_units(row["aligned_frame_path"], device = device))
        au_intensities.append(get_au_intensities(row["aligned_frame_path"], device = device))
        facial_expression.append(get_facial_expression(row["aligned_frame_path"], device = device))
    frames_df["detected_aus"] = detected_aus
    frames_df["au_intensities"] = au_intensities
    frames_df["facial_expression"] = facial_expression
    frames_df = frames_df.join(pd.json_normalize(frames_df['detected_aus'])).drop('detected_aus', axis='columns')
    frames_df = frames_df.join(pd.json_normalize(frames_df['au_intensities'])).drop('au_intensities', axis='columns')
    return frames_df

def save_facial_attributes_video(video_path, 
                            output_save_path = "video_results.csv", 
                            temp_dir="./tmp", 
                            device="cpu"):
    frames_df = get_facial_attributes_video(video_path,
                                            temp_dir,
                                            device)
    save_path = uniquify_file(output_save_path)
    frames_df.to_csv(save_path)
    print(f"Facial attributes of the video saved to {save_path}")
    return save_path

def save_facial_attributes_image(image_path, 
                                 output_save_path = "image_results.csv", 
                                 temp_dir="./tmp", 
                                 device="cpu"):
    attr_dict = get_facial_attributes_image(image_path, temp_dir, device)
    for k, v in attr_dict.items():
        attr_dict[k] = [v]
    attr_df = pd.DataFrame(attr_dict)
    attr_df = attr_df.join(pd.json_normalize(attr_df['detected_aus'])).drop('detected_aus', axis='columns')
    attr_df = attr_df.join(pd.json_normalize(attr_df['au_intensities'])).drop('au_intensities', axis='columns')
    save_path = uniquify_file(output_save_path)
    attr_df.to_csv(save_path)
    print(f"Facial attributes of the image saved to {save_path}")
    return save_path

def get_facial_attributes(file_path, output_save_path=None, temp_dir="./tmp", device="cpu"):
    file_type = check_file_type(file_path)
    if file_type == "image":
        if output_save_path is None:
            return get_facial_attributes_image(file_path, temp_dir=temp_dir, device=device)
        else:
            try:
                return save_facial_attributes_image(file_path, output_save_path=output_save_path, temp_dir=temp_dir, device=device)
            except Exception as e:
                print(e)
                print("Some error in saving the results.")
    elif file_type == "video":
        if output_save_path is None:
            return get_facial_attributes_video(file_path, temp_dir=temp_dir, device=device)
        else:
            try:
                return save_facial_attributes_video(file_path, output_save_path=output_save_path, temp_dir=temp_dir, device=device)
            except Exception as e:
                print(e)
                print("Some error in saving the results.")
                return False



