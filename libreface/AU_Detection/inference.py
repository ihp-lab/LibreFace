import argparse
import numpy as np
import pandas as pd
import random
import torch
import yaml

from libreface.AU_Detection.solver_inference_image import solver_in_domain_image

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

def format_output(out_dict):
    new_dict = {}
    for k, v in out_dict.items():
        new_dict[f"au_{k}"] = v
    return new_dict

def format_output_video(aus, detected_aus):
    df_columns = [f"au_{k}" for k in aus]
    detected_au_df = pd.DataFrame(detected_aus, columns = df_columns)
    # detected_au_df["frame_idx"] = list(range(len(detected_au_df.index)))
    return detected_au_df

def detect_action_units(image_path:str, 
                        device:str = "cpu",
                        weights_download_dir = "./weights_libreface")->dict:
    """This method takes an image path as input and detects the action 
    units present in the image. Right now the action units covered are 

    Args:
        image_path (str): Path to the input image
        device (str, optional): Device to be used for pytorch inference. Can be "cpu" or "cuda". Defaults to "cpu".

    Returns:
        dict: Dictionary containing the different action unit numbers as keys and having values `1` for the action
            units that are detected and `0` for the ones that are not detected. 
    """
    opts = ConfigObject({'seed': 0,
                        'data_root': '',
                        'ckpt_path': f'{weights_download_dir}/AU_Detection/weights/resnet.pt',
                        'weights_download_id': '17v_vxQ09upLG3Yh0Zlx12rpblP7uoA8x',
                        'data': 'BP4D',
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
                        'threshold': 0.0,
                        'batch_size': 256,
                        'learning_rate': '3e-5',
                        'weight_decay': '1e-4',
                        'loss': 'unweighted',
                        'clip': 1.0,
                        'when': 10,
                        'patience': 5,
                        'fm_distillation': False,
                        'device': 'cpu'
                    })

    #set seed
    set_seed(opts.seed)

    opts.device = device
    # opts.ckpt_path = model_path
    # print(f"Using device: {opts.device} for inference...")
    solver = solver_in_domain_image(opts).to(device)
    detected_aus = solver.run(image_path)
    return format_output(detected_aus)

def detect_action_units_video(aligned_frames_path_list:list, 
                        device:str = "cpu",
                        batch_size:int = 256,
                        num_workers:int = 2,
                        weights_download_dir = "./weights_libreface")->dict:
    
    opts = ConfigObject({'seed': 0,
                        'data_root': '',
                        'ckpt_path': f'{weights_download_dir}/AU_Detection/weights/resnet.pt',
                        'weights_download_id': '17v_vxQ09upLG3Yh0Zlx12rpblP7uoA8x',
                        'data': 'BP4D',
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
                        'threshold': 0.0,
                        'batch_size': batch_size,
                        'learning_rate': '3e-5',
                        'weight_decay': '1e-4',
                        'loss': 'unweighted',
                        'clip': 1.0,
                        'when': 10,
                        'patience': 5,
                        'fm_distillation': False,
                        'device': 'cpu'
                    })

    #set seed
    set_seed(opts.seed)

    opts.device = device
    solver = solver_in_domain_image(opts).to(device)
    aus, detected_aus = solver.run_video(aligned_frames_path_list)
    return format_output_video(aus, detected_aus)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    # parser.add_argument("--model_path", type=str, default="./libreface/AU_Detection/fm_distillation_all/BP4D/all/resnet.pt")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    detected_aus = detect_action_units(image_path = args.image_path,
                                    #    model_path = args.model_path,
                                       device = args.device)
    print(f"Detected action units - {detected_aus}")
    

if __name__ == "__main__":
    main()