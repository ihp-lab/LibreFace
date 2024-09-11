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

def detect_action_units(image_path, device="cuda"):
    
    # Path to the YAML config file
    config_path = './libreface/AU_Detection/config_au_detection.yaml'

    # Load the configuration from YAML
    config = load_config(config_path)
    opts = ConfigObject(config)

    #set seed
    set_seed(opts.seed)

    opts.device = device
    print(f"Using device: {opts.device} for inference...")
    solver = solver_in_domain_image(opts).to(device)

    detected_aus = solver.run(image_path)
    return detected_aus

def set_seed(seed):
    # Reproducibility
    torch.manual_seed(seed)

    random.seed(seed)
    np.random.seed(seed)

# Function to set argparse defaults from the YAML config
def set_defaults_from_config(parser, config):
    parser.set_defaults(**config)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    detected_aus = detect_action_units(image_path = args.image_path, device = args.device)
    print(f"Detected action units - {detected_aus}")
    

if __name__ == "__main__":
    main()