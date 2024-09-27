import pandas as pd
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

from libreface.detect_mediapipe_image import *
from libreface.AU_Detection.inference import detect_action_units, detect_action_units_video
from libreface.AU_Recognition.inference import get_au_intensities, get_au_intensities_and_detect_aus, get_au_intensities_and_detect_aus_video, get_au_intensities_video
from libreface.Facial_Expression_Recognition.inference import get_facial_expression, get_facial_expression_video
from libreface.utils import get_frames_from_video_ffmpeg, uniquify_file, check_file_type

def get_facial_attributes_image(image_path:str, 
                                model_choice:str="joint_au_detection_intensity_estimator", 
                                temp_dir:str="./tmp", 
                                device:str="cpu",
                                weights_download_dir:str = "./weights_libreface")->dict:
    """Get facial attributes for an image. This function reads an image and returns a dictionary containing
    some detected facial action units and expressions.

    Args:
        image_path (str): Input image path.
        model_choice (str, optional): Model to use when doing predictions. Defaults to "joint_au_detection_intensity_estimator".
        temp_dir (str, optional): Path where the temporary aligned image, facial landmarks 
        and landmark annotated image will be stored. Defaults to "./tmp".
        device (str, optional): device to be used for inference. Can be "cpu" or "cuda". Defaults to "cpu".
        weights_download_dir(str, optional): directory where you want to download and save the model weights.

    Returns:
        dict: dictionary containing the following keys
            input_image_path - copied from the image_path
            aligned_image_path - path to the aligned image, i.e. image with the only the face of the person cropped from original image
            detected_action_units - dictionary of detected action units. Units which are detected have value of 1, else 0.
            au_intensities - dictionary of action unit intensities, with each intensity in the range (0, 5)
            facial_expression - detected facial expression. Can be one from ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
    """
    
    print(f"Using device: {device} for inference...")
    aligned_image_path, headpose, landmarks_2d = get_aligned_image(image_path, temp_dir=temp_dir, verbose=True)
    if model_choice == "joint_au_detection_intensity_estimator":
        detected_aus, au_intensities = get_au_intensities_and_detect_aus(aligned_image_path, device=device, weights_download_dir=weights_download_dir)
    elif model_choice == "separate_prediction_heads":
        detected_aus = detect_action_units(aligned_image_path, device = device, weights_download_dir=weights_download_dir)
        au_intensities = get_au_intensities(aligned_image_path, device = device, weights_download_dir=weights_download_dir)
    else:
        print(f"Undefined model_choice = {model_choice} for get_facial_attributes_image()")
        raise NotImplementedError
    facial_expression = get_facial_expression(aligned_image_path, device = device, weights_download_dir=weights_download_dir)
    return_dict =  {
            "detected_aus": detected_aus,
            "au_intensities": au_intensities,
            "facial_expression": facial_expression}
    
    return_dict = {**return_dict, **headpose, **landmarks_2d}

    return return_dict

def get_facial_attributes_video(video_path, 
                                model_choice:str="joint_au_detection_intensity_estimator", 
                                temp_dir="./tmp", 
                                device="cpu",
                                batch_size = 256,
                                num_workers = 2,
                                weights_download_dir:str = "./weights_libreface"):
    print(f"Using device: {device} for inference...")
    
    frames_df = get_frames_from_video_ffmpeg(video_path, temp_dir=temp_dir)
    cur_video_name = ".".join(video_path.split("/")[-1].split(".")[:-1])
    aligned_frames_path_list, headpose_list, landmarks_3d_list = get_aligned_video_frames(frames_df, temp_dir=os.path.join(temp_dir, cur_video_name))
    # frames_df["aligned_frame_path"] = aligned_frames_path_list
    frames_df = frames_df.drop("path_to_frame", axis=1)
    frames_df["headpose"] = headpose_list
    frames_df["landmarks_3d"] = landmarks_3d_list
     

    frames_df = frames_df.join(pd.json_normalize(frames_df['headpose'])).drop('headpose', axis='columns')
    frames_df = frames_df.join(pd.json_normalize(frames_df['landmarks_3d'])).drop('landmarks_3d', axis='columns')

    detected_aus, au_intensities, facial_expression = [], [], []
    
    if model_choice == "joint_au_detection_intensity_estimator":
        detected_aus, au_intensities = get_au_intensities_and_detect_aus_video(aligned_frames_path_list, 
                                                                           device = device, 
                                                                           batch_size=batch_size,
                                                                           num_workers=num_workers,
                                                                           weights_download_dir=weights_download_dir)
    elif model_choice == "separate_prediction_heads":
        detected_aus = detect_action_units_video(aligned_frames_path_list, 
                                                device=device,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                weights_download_dir=weights_download_dir)
        au_intensities = get_au_intensities_video(aligned_frames_path_list, 
                                                device=device,
                                                batch_size=batch_size,
                                                num_workers=num_workers,
                                                weights_download_dir=weights_download_dir)
    else:
        print(f"Undefined model_choice = {model_choice} for get_facial_attributes_video()")
        raise NotImplementedError
    
    facial_expression = get_facial_expression_video(aligned_frames_path_list, 
                                                    device = device, 
                                                    batch_size=batch_size,
                                                    weights_download_dir=weights_download_dir)
    

    frames_df = frames_df.join(detected_aus)
    frames_df = frames_df.join(au_intensities)
    frames_df = frames_df.join(facial_expression)
    return frames_df

def save_facial_attributes_video(video_path, 
                            output_save_path = "video_results.csv", 
                            model_choice:str="joint_au_detection_intensity_estimator",
                            temp_dir="./tmp", 
                            device="cpu",
                            batch_size = 256,
                            num_workers = 2,
                            weights_download_dir:str = "./weights_libreface"):
    frames_df = get_facial_attributes_video(video_path,
                                            model_choice=model_choice,
                                            temp_dir=temp_dir,
                                            device=device, 
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            weights_download_dir=weights_download_dir)
    save_path = uniquify_file(output_save_path)
    frames_df.to_csv(save_path, index=False)
    print(f"Facial attributes of the video saved to {save_path}")
    return save_path

def save_facial_attributes_image(image_path, 
                                 output_save_path = "image_results.csv", 
                                 model_choice:str="joint_au_detection_intensity_estimator",
                                 temp_dir="./tmp", 
                                 device="cpu",
                                 weights_download_dir:str = "./weights_libreface"):
    attr_dict = get_facial_attributes_image(image_path, 
                                            model_choice=model_choice, 
                                            temp_dir=temp_dir, 
                                            device=device,
                                            weights_download_dir=weights_download_dir)
    for k, v in attr_dict.items():
        attr_dict[k] = [v]
    attr_df = pd.DataFrame(attr_dict)
    attr_df = attr_df.join(pd.json_normalize(attr_df['detected_aus'])).drop('detected_aus', axis='columns')
    attr_df = attr_df.join(pd.json_normalize(attr_df['au_intensities'])).drop('au_intensities', axis='columns')
    attr_df.index.name = 'frame_idx'
    save_path = uniquify_file(output_save_path)
    attr_df.to_csv(save_path, index=False)
    print(f"Facial attributes of the image saved to {save_path}")
    return save_path

def get_facial_attributes(file_path, 
                          output_save_path=None, 
                          model_choice:str="joint_au_detection_intensity_estimator",
                          temp_dir="./tmp", 
                          device="cpu",
                          batch_size = 256,
                          num_workers = 2,
                          weights_download_dir:str = "./weights_libreface"):
    file_type = check_file_type(file_path)
    if file_type == "image":
        if output_save_path is None:
            return get_facial_attributes_image(file_path, model_choice=model_choice, 
                                               temp_dir=temp_dir, device=device, weights_download_dir=weights_download_dir)
        else:
            try:
                return save_facial_attributes_image(file_path, output_save_path=output_save_path, 
                                                    model_choice=model_choice, temp_dir=temp_dir, 
                                                    device=device, weights_download_dir=weights_download_dir)
            except Exception as e:
                print(e)
                print("Some error in saving the results.")
    elif file_type == "video":
        if output_save_path is None:
            return get_facial_attributes_video(file_path, model_choice=model_choice, 
                                               temp_dir=temp_dir, device=device, 
                                               batch_size=batch_size, 
                                               num_workers=num_workers, weights_download_dir=weights_download_dir)
        else:
            try:
                return save_facial_attributes_video(file_path, output_save_path=output_save_path, 
                                                    model_choice=model_choice, temp_dir=temp_dir, 
                                                    device=device, batch_size=batch_size, 
                                                    num_workers=num_workers, weights_download_dir=weights_download_dir)
            except Exception as e:
                print(e)
                print("Some error in saving the results.")
                return False



