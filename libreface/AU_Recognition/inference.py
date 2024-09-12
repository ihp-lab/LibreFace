import argparse
import numpy as np
import random
import torch
import yaml

from libreface.AU_Recognition.solver_inference_image import solver_inference_image

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

def get_au_intensities(image_path, model_path, device="cuda"):
    
    # # Path to the YAML config file
    # config_path = './libreface/AU_Recognition/config_au_recognition.yaml'

    # # Load the configuration from YAML
    # config = load_config(config_path)
    opts = ConfigObject({'seed': 0,
                        'data_root': '',
                        'ckpt_path': './libreface/AU_Recognition/resnet_disfa_all/DISFA/all/resnet.pt',
                        'data': 'DISFA',
                        'fold': 'all',
                        'num_workers': 0,
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
                        'device': 'cuda'})

    #set seed
    set_seed(opts.seed)

    opts.device = device
    opts.ckpt_path = model_path
    print(f"Using device: {opts.device} for inference...")

    solver = solver_inference_image(opts).to(device)

    au_intensities = solver.run(image_path)
    return au_intensities

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add arguments (same as your original argparse setup)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="./libreface/AU_Recognition/resnet_disfa_all/DISFA/all/resnet.pt")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    au_intensities = get_au_intensities(args.image_path, 
                                        model_path = args.model_path,
                                        device = args.device)
    print(f"Predicted action unit intensities (On scale 0-5): {au_intensities}")
    


if __name__ == "__main__":
    main()

