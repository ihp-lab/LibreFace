import argparse
import yaml

from src.AU_Recognition.solver_inference_image import solver_inference_image
from src.AU_Recognition.solver_inference import solver_inference
from src.AU_Recognition.utils import set_seed

class ConfigObject:
    def __init__(self, config_dict):
        # Set each key-value pair in the dictionary as an attribute
        for key, value in config_dict.items():
            setattr(self, key, value)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def au_recognition_inference_image(image_path):
    
    # Path to the YAML config file
    config_path = './src/AU_Recognition/config_au_recognition.yaml'

    # Load the configuration from YAML
    config = load_config(config_path)
    opts = ConfigObject(config)

    solver = solver_inference_image(opts).cuda()

    solver.run(image_path)

# Function to set argparse defaults from the YAML config
def set_defaults_from_config(parser, config):
    parser.set_defaults(**config)

def main():
    # Path to the YAML config file
    config_path = './src/AU_Recognition/config_au_recognition.yaml'

    # Load the configuration from YAML
    config = load_config(config_path)

    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Add arguments (same as your original argparse setup)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--image_inference", action="store_true", default=True)
    # storage
    parser.add_argument('--data_root', type=str, default='/home/ICT2000/dchang/TAC_project/Face_Heatmap/data')
    parser.add_argument('--ckpt_path', type=str, default='./src/AU_Recognition/resnet_disfa_all')

    # data
    parser.add_argument('--data', type=str, choices=['BP4D', 'DISFA'])
    parser.add_argument('--fold', type=str, choices=['0', '1', '2', '3', '4','all'])
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--crop_size', type=int)
    parser.add_argument('--num_labels', type=int)
    parser.add_argument('--sigma', type=float)

    # model
    parser.add_argument('--model_name', type=str, choices=['resnet_heatmap','resnet','swin','mae','emotionnet_mae','gh_feat'])
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--hidden_dim', type=int) 
    parser.add_argument('--half_precision', action='store_true')

    # training
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--interval', type=int)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--loss', type=str)
    parser.add_argument('--clip', type=int)
    parser.add_argument('--when', type=int, help='when to decay learning rate')
    parser.add_argument('--patience', type=int, help='early stopping')
    parser.add_argument('--fm_distillation', action='store_true')

    # device
    parser.add_argument('--device', type=str, choices=['cpu','cuda'])

    # Set the defaults from the YAML configuration
    set_defaults_from_config(parser, config)

    opts = parser.parse_args()

    # Fix random seed
    set_seed(opts.seed)

    if opts.image_inference:
        au_recognition_inference_image("/home/achaubey/Desktop/projects/data/DISFA/output/aligned_images/LeftVideoSN011_comp/LeftVideoSN011_comp_0001.png")
    else:
        solver = solver_inference(opts).cuda()

        solver.run()


if __name__ == "__main__":
    main()

