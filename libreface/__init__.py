from libreface.detect_mediapipe_image import *
from libreface.AU_Detection.inference import detect_action_units
from libreface.AU_Recognition.inference import get_au_intensities
from libreface.Facial_Expression_Recognition.inference import get_facial_expression
from libreface.utils import get_frames_from_video_decord

def get_facial_attributes_image(image_path, temp_dir="./tmp", device="cpu"):
    aligned_image_path = get_aligned_image(image_path, temp_dir=temp_dir)
    detected_aus = detect_action_units(aligned_image_path, device = device)
    au_intensities = get_au_intensities(aligned_image_path, device = device)
    facial_expression = get_facial_expression(aligned_image_path, device = device)
    return {"detected_aus": detected_aus,
            "au_intensities": au_intensities,
            "facial_expression": facial_expression}

def get_facial_attributes_video(video_path, 
                                output_save_path, 
                                temp_dir="./tmp", 
                                device="cpu"):
    frames_df = get_frames_from_video_decord(video_path, temp_dir=temp_dir)
    cur_video_name = ".".join(video_path.split("/")[-1].split(".")[:-1])
    aligned_frames_path_list = get_aligned_video_frames(frames_df, temp_dir=os.path.join(temp_dir, cur_video_name))
    frames_df["aligned_frame_path"] = aligned_frames_path_list

    detected_aus, au_intensities, facial_expression = [], [], []
    for _, row in frames_df.iterrows():
        detected_aus.append(detect_action_units(row["aligned_frame_path"], device = device))
        au_intensities.append(get_au_intensities(row["aligned_frame_path"], device = device))
        facial_expression.append(get_facial_expression(row["aligned_frame_path"], device = device))
    frames_df["detected_aus"] = detected_aus
    frames_df["au_intensities"] = au_intensities
    frames_df["facial_expression"] = facial_expression
    frames_df.to_pickle(output_save_path)

