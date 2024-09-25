import argparse
import numpy as np
import pandas as pd
import random
import torch
import yaml

from libreface.AU_Recognition.solver_inference_image import solver_inference_image
from libreface.AU_Recognition.solver_inference_combine import solver_inference_image_task_combine

class ConfigObject:
    def __init__(self, config_dict):
        # Set each key-value pair in the dictionary as an attribute
        for key, value in config_dict.items():
            setattr(self, key, value)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed):
    # Reproducibility
    torch.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

def format_output(out_dict, task = "au_recognition"):
    new_dict = {}
    for k, v in out_dict.items():
        if task == "au_recognition":
            new_dict[f"au_{k}_intensity"] = round(v, 3)
        elif task == "au_detection":
            new_dict[f"au_{k}"] = v
        else:
            raise NotImplementedError(f"format_output() not defined for the task - {task}")
    return new_dict

def format_output_video(aus, out_dict, task = "au_recognition"):
    if task == "au_recognition":
        df_columns = [f"au_{k}_intensity" for k in aus]
    elif task == "au_detection":
        df_columns = [f"au_{k}" for k in aus]
    else:
        raise NotImplementedError(f"format_output() not defined for the task - {task}")
    au_df = pd.DataFrame(out_dict, columns = df_columns)
    return au_df

def get_au_intensities_and_detect_aus(image_path, device="cpu",
                                      weights_download_dir="./weights_libreface"):
    opts = ConfigObject({'seed':0,
                        'ckpt_path': f'{weights_download_dir}/AU_Recognition/weights/combined_resnet.pt',
                        'weights_download_id':"1CbnBr8OBt8Wb73sL1ENcrtrWAFWSSRv0", 
                        'image_inference': False,
                        'au_recognition_data_root': '',
                        'au_recognition_data': 'DISFA',
                        'au_detection_data_root': '',
                        'au_detection_data': 'BP4D',
                        'fer_train_csv': 'training_filtered.csv',
                        'fer_test_csv': 'validation_filtered.csv',
                        'fer_data_root': '',
                        'fer_data': 'AffectNet',
                        'fold': 'all',
                        'image_size': 256,
                        'crop_size': 224,
                        'au_recognition_num_labels': 12,
                        'au_detection_num_labels': 12,
                        'fer_num_labels': 8,
                        'sigma': 10.0,
                        'jitter': False,
                        'copy_classifier': False,
                        'model_name': 'resnet',
                        'dropout': 0.1,
                        'ffhq_pretrain': '',
                        'hidden_dim': 128,
                        'fm_distillation': False,
                        'num_epochs': 30,
                        'interval': 500,
                        'threshold': 0,
                        'batch_size': 256,
                        'learning_rate': 3e-5,
                        'weight_decay': 1e-4,
                        'clip': 1.0,
                        'when': 10,
                        'patience': 5,
                        'device': 'cuda'
                    })
    
    #set seed
    set_seed(opts.seed)

    opts.device = device
    # print(f"Using device: {opts.device} for inference...")

    solver = solver_inference_image_task_combine(opts).to(device)

    detected_aus = solver.run(image_path, task = "au_detection")
    au_intensities = solver.run(image_path, task = "au_recognition")
    return format_output(detected_aus, task="au_detection"), format_output(au_intensities, task = "au_recognition")

def get_au_intensities_and_detect_aus_video(aligned_frames_path, device="cpu",
                                            batch_size = 256, num_workers=2,
                                      weights_download_dir="./weights_libreface"):
    opts = ConfigObject({'seed':0,
                        'ckpt_path': f'{weights_download_dir}/AU_Recognition/weights/combined_resnet.pt',
                        'weights_download_id':"1CbnBr8OBt8Wb73sL1ENcrtrWAFWSSRv0", 
                        'image_inference': False,
                        'au_recognition_data_root': '',
                        'au_recognition_data': 'DISFA',
                        'au_detection_data_root': '',
                        'au_detection_data': 'BP4D',
                        'fer_train_csv': 'training_filtered.csv',
                        'fer_test_csv': 'validation_filtered.csv',
                        'fer_data_root': '',
                        'fer_data': 'AffectNet',
                        'fold': 'all',
                        'num_workers': num_workers,
                        'image_size': 256,
                        'crop_size': 224,
                        'au_recognition_num_labels': 12,
                        'au_detection_num_labels': 12,
                        'fer_num_labels': 8,
                        'sigma': 10.0,
                        'jitter': False,
                        'copy_classifier': False,
                        'model_name': 'resnet',
                        'dropout': 0.1,
                        'ffhq_pretrain': '',
                        'hidden_dim': 128,
                        'fm_distillation': False,
                        'num_epochs': 30,
                        'interval': 500,
                        'threshold': 0,
                        'batch_size': batch_size,
                        'learning_rate': 3e-5,
                        'weight_decay': 1e-4,
                        'clip': 1.0,
                        'when': 10,
                        'patience': 5,
                        'device': 'cuda'
                    })
    
    #set seed
    set_seed(opts.seed)

    opts.device = device
    # print(f"Using device: {opts.device} for inference...")

    solver = solver_inference_image_task_combine(opts).to(device)

    det_aus, detected_aus = solver.run_video(aligned_frames_path, task = "au_detection")
    reco_aus, au_intensities = solver.run_video(aligned_frames_path, task = "au_recognition")

    return format_output_video(det_aus, detected_aus, task="au_detection"), format_output_video(reco_aus, au_intensities, task = "au_recognition")

def get_au_intensities(image_path, device="cpu",
                        weights_download_dir = "./weights_libreface"):
    
    # # Path to the YAML config file
    # config_path = './libreface/AU_Recognition/config_au_recognition.yaml'

    # # Load the configuration from YAML
    # config = load_config(config_path)
    opts = ConfigObject({'seed': 0,
                        'data_root': '',
                        'ckpt_path': f'{weights_download_dir}/AU_Recognition/weights/resnet.pt',
                        'weights_download_id': '14qEnWRew2snhdMdOVyqKFJ5rq5VZrfAX',
                        'data': 'DISFA',
                        'fold': 'all',
                        'image_size': 256,
                        'crop_size': 224,
                        'num_labels': 12,
                        'sigma': 10.0,
                        'model_name': 'resnet',
                        'dropout': 0.1,
                        'hidden_dim': 128,
                        'half_precision': False,
                        'num_epochs': 30,
                        'interval': 500,
                        'threshold': 0,
                        'batch_size': 256,
                        'learning_rate': '3e-5',
                        'weight_decay': '1e-4',
                        'loss': 'unweighted',
                        'clip': 1.0,
                        'when': 10,
                        'patience': 5,
                        'fm_distillation': False,
                        'device': 'cpu'})

    #set seed
    set_seed(opts.seed)

    opts.device = device
    # print(f"Using device: {opts.device} for inference...")

    solver = solver_inference_image(opts).to(device)

    au_intensities = solver.run(image_path)
    return format_output(au_intensities)

def get_au_intensities_video(aligned_frames_path, device="cpu",
                             batch_size = 256, num_workers = 2,
                        weights_download_dir = "./weights_libreface"):
    
    # # Path to the YAML config file
    # config_path = './libreface/AU_Recognition/config_au_recognition.yaml'

    # # Load the configuration from YAML
    # config = load_config(config_path)
    opts = ConfigObject({'seed': 0,
                        'data_root': '',
                        'ckpt_path': f'{weights_download_dir}/AU_Recognition/weights/resnet.pt',
                        'weights_download_id': '14qEnWRew2snhdMdOVyqKFJ5rq5VZrfAX',
                        'data': 'DISFA',
                        'fold': 'all',
                        'num_workers': num_workers,
                        'image_size': 256,
                        'crop_size': 224,
                        'num_labels': 12,
                        'sigma': 10.0,
                        'model_name': 'resnet',
                        'dropout': 0.1,
                        'hidden_dim': 128,
                        'half_precision': False,
                        'num_epochs': 30,
                        'interval': 500,
                        'threshold': 0,
                        'batch_size': batch_size,
                        'learning_rate': '3e-5',
                        'weight_decay': '1e-4',
                        'loss': 'unweighted',
                        'clip': 1.0,
                        'when': 10,
                        'patience': 5,
                        'fm_distillation': False,
                        'device': 'cpu'})

    #set seed
    set_seed(opts.seed)

    opts.device = device
    # print(f"Using device: {opts.device} for inference...")

    solver = solver_inference_image(opts).to(device)

    aus, au_intensities = solver.run_video(aligned_frames_path)
    return format_output_video(aus, au_intensities)

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add arguments (same as your original argparse setup)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    au_intensities = get_au_intensities(args.image_path, 
                                        device = args.device)
    print(f"Predicted action unit intensities (On scale 0-5): {au_intensities}")
    


if __name__ == "__main__":
    main()

