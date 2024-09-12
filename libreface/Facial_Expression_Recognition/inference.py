import argparse
import numpy as np
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

def get_facial_expression(image_path, model_path, device = "cuda"):
    # # Path to the YAML config file
    # config_path = './libreface/Facial_Expression_Recognition/config_fer.yaml'

    # # Load the configuration from YAML
    # config = load_config(config_path)
    opts = ConfigObject({'seed': 0,
                        'train_csv': 'training_filtered.csv',
                        'test_csv': 'validation_filtered.csv',
                        'data_root': '',
                        'ckpt_path': './libreface/Facial_Expression_Recognition/checkpoints_fm_resnet/AffectNet/resnet.pt',
                        'data': 'AffectNet',
                        'num_workers': 8,
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
                        'device': 'cuda'
                    })           

    #set seed
    set_seed(opts.seed)

    opts.device = device
    opts.ckpt_path = model_path
    print(f"Using device: {opts.device} for inference...")

    solver = solver_inference_image(opts).to(device)

    facial_expression = solver.run(image_path)
    return facial_expression

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add arguments (same as your original argparse setup)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="./libreface/Facial_Expression_Recognition/checkpoints_fm_resnet/AffectNet/resnet.pt")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    facial_expression = get_facial_expression(args.image_path, 
                                              model_path = args.model_path, 
                                              device = args.device)
    print(f"Predicted facial expression : {facial_expression}")
    
if __name__ == "__main__":
    main()
