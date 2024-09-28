import argparse
import numpy as np
import pandas as pd
import random
import torch
import yaml

from libreface.Facial_Expression_Recognition.solver_inference_image import solver_inference_image

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

def facial_expr_idx_to_class(fe_idx):
    idx_to_fe = {0: "Neutral",
                 1: "Happiness",
                 2: "Sadness",
                 3: "Surprise",
                 4: "Fear",
                 5: "Disgust",
                 6: "Anger",
                 7: "Contempt"}
    return idx_to_fe[fe_idx]

def facial_expr_idx_to_class_video(fe_idx_list):
    return pd.DataFrame([[facial_expr_idx_to_class(idx)] for idx in fe_idx_list], columns = ["facial_expression"])

def get_facial_expression(image_path, device = "cpu",
                        weights_download_dir = "./weights_libreface"):
    opts = ConfigObject({'seed': 0,
                        'train_csv': 'training_filtered.csv',
                        'test_csv': 'validation_filtered.csv',
                        'data_root': '',
                        'ckpt_path': f'{weights_download_dir}/Facial_Expression_Recognition/weights/resnet.pt',
                        'weights_download_id': '1PeoPj8rga4vU2nuh_PciyX3HqaXp6LP7',
                        'data': 'AffectNet',
                        'image_size': 224,
                        'num_labels': 8,
                        'dropout': 0.1,
                        'hidden_dim': 128,
                        'sigma': 10.0,
                        'student_model_name': 'resnet',
                        'student_model_choices': [
                            'resnet_heatmap', 
                            'resnet', 
                            'swin', 
                            'mae', 
                            'emotionnet_mae', 
                            'gh_feat'
                        ],
                        'alpha': 1.0,
                        'T': 1.0,
                        'fm_distillation': True,
                        'grad': True,
                        'interval': 500,
                        'threshold': 0.0,
                        'loss': 'unweighted',
                        'num_epochs': 50,
                        'batch_size': 256,
                        'learning_rate': '3e-5',
                        'weight_decay': '1e-4',
                        'clip': 1.0,
                        'when': 10,
                        'patience': 10,
                        'device': 'cpu'
                    })           

    #set seed
    set_seed(opts.seed)

    opts.device = device
    # print(f"Using device: {opts.device} for inference...")

    solver = solver_inference_image(opts).to(device)

    facial_expression = solver.run(image_path)
    return facial_expr_idx_to_class(facial_expression)

def get_facial_expression_video(aligned_frames_path, device = "cpu",
                                batch_size = 256, num_workers = 2,
                        weights_download_dir = "./weights_libreface"):
    opts = ConfigObject({'seed': 0,
                        'train_csv': 'training_filtered.csv',
                        'test_csv': 'validation_filtered.csv',
                        'data_root': '',
                        'ckpt_path': f'{weights_download_dir}/Facial_Expression_Recognition/weights/resnet.pt',
                        'weights_download_id': '1PeoPj8rga4vU2nuh_PciyX3HqaXp6LP7',
                        'data': 'AffectNet',
                        'num_workers': num_workers,
                        'image_size': 224,
                        'num_labels': 8,
                        'dropout': 0.1,
                        'hidden_dim': 128,
                        'sigma': 10.0,
                        'student_model_name': 'resnet',
                        'student_model_choices': [
                            'resnet_heatmap', 
                            'resnet', 
                            'swin', 
                            'mae', 
                            'emotionnet_mae', 
                            'gh_feat'
                        ],
                        'alpha': 1.0,
                        'T': 1.0,
                        'fm_distillation': True,
                        'grad': True,
                        'interval': 500,
                        'threshold': 0.0,
                        'loss': 'unweighted',
                        'num_epochs': 50,
                        'batch_size': batch_size,
                        'learning_rate': '3e-5',
                        'weight_decay': '1e-4',
                        'clip': 1.0,
                        'when': 10,
                        'patience': 10,
                        'device': 'cpu'
                    })           

    #set seed
    set_seed(opts.seed)

    opts.device = device
    # print(f"Using device: {opts.device} for inference...")

    solver = solver_inference_image(opts).to(device)

    facial_expression = solver.run_video(aligned_frames_path)
    return facial_expr_idx_to_class_video(facial_expression)

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add arguments (same as your original argparse setup)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    facial_expression = get_facial_expression(args.image_path, 
                                              device = args.device)
    print(f"Predicted facial expression : {facial_expression}")
    
if __name__ == "__main__":
    main()
