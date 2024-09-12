import argparse
import numpy as np
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

def detect_action_units(image_path, model_path, device="cpu"):
    
    # Path to the YAML config file
    # config_path = './libreface/AU_Detection/config_au_detection.yaml'

    # Load the configuration from YAML
    # config = load_config(config_path)
    opts = ConfigObject({'seed': 0,
                        'data_root': '/home/ICT2000/dchang/TAC_project/data',
                        'ckpt_path': './libreface/AU_Detection/fm_distillation_all/BP4D/all/resnet.pt',
                        'data': 'BP4D',
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
    opts.ckpt_path = model_path
    print(f"Using device: {opts.device} for inference...")
    solver = solver_in_domain_image(opts).to(device)

    detected_aus = solver.run(image_path)
    return detected_aus

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="./libreface/AU_Detection/fm_distillation_all/BP4D/all/resnet.pt")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    detected_aus = detect_action_units(image_path = args.image_path,
                                       model_path = args.model_path,
                                       device = args.device)
    print(f"Detected action units - {detected_aus}")
    

if __name__ == "__main__":
    main()