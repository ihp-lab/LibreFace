import argparse
import numpy as np
import pandas as pd
import random
import torch

from libreface.gaze_estimation.solver_inference_image import solver_gaze_image, GAZE_FEAT_DIM


class ConfigObject:
    def __init__(self, config_dict):
        # Set each key-value pair in the dictionary as an attribute
        for key, value in config_dict.items():
            setattr(self, key, value)


def set_seed(seed):
    # Reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def format_output(out_dict):
    return {
        "gaze_yaw": out_dict["yaw"],
        "gaze_pitch": out_dict["pitch"],
    }


def format_output_video(preds):
    detected_gaze_df = pd.DataFrame(preds, columns=["gaze_yaw", "gaze_pitch"])
    return detected_gaze_df


def estimate_gaze(image_path: str,
                  device: str = "cpu",
                  weights_download_dir: str = "./weights_libreface") -> dict:
    """Takes an aligned image path as input and estimates the gaze angles.

    Args:
        image_path (str): Path to the aligned input image.
        device (str, optional): Device for PyTorch inference. Can be "cpu" or "cuda". Defaults to "cpu".
        weights_download_dir (str, optional): Directory to download and cache model weights.

    Returns:
        dict: Dictionary with keys "gaze_yaw" and "gaze_pitch" (both in degrees).
    """
    opts = ConfigObject({
        'seed': 0,
        'data_root': '',
        'ckpt_path': f'{weights_download_dir}/gaze_estimation/weights/mlp.pt',
        'weights_download_id': 'GAZE_WEIGHTS_DOWNLOAD_ID',  # replace with actual GDrive ID
        'data': 'Gaze360',
        'fold': 'all',
        'num_labels': 2,
        'model_name': 'mlp',
        'mlp_input_size': GAZE_FEAT_DIM,
        'dropout': 0.1,
        'hidden_dim': 128,
        'half_precision': False,
        'batch_size': 256,
        'num_workers': 2,
        'device': 'cpu',
    })

    set_seed(opts.seed)
    opts.device = device

    solver = solver_gaze_image(opts).to(device)
    result = solver.run(image_path)
    return format_output(result)


def estimate_gaze_video(aligned_frames_path_list: list,
                        device: str = "cpu",
                        batch_size: int = 256,
                        num_workers: int = 2,
                        weights_download_dir: str = "./weights_libreface") -> pd.DataFrame:
    """Takes a list of aligned image paths and estimates gaze angles per frame.

    Args:
        aligned_frames_path_list (list): List of aligned image paths, one per frame.
        device (str, optional): Device for PyTorch inference. Can be "cpu" or "cuda". Defaults to "cpu".
        batch_size (int, optional): Batch size for video inference. Defaults to 256.
        num_workers (int, optional): DataLoader worker count. Defaults to 2.
        weights_download_dir (str, optional): Directory to download and cache model weights.

    Returns:
        pd.DataFrame: DataFrame with columns "gaze_yaw" and "gaze_pitch", one row per frame.
    """
    opts = ConfigObject({
        'seed': 0,
        'data_root': '',
        'ckpt_path': f'{weights_download_dir}/gaze_estimation/weights/mlp.pt',
        'weights_download_id': 'GAZE_WEIGHTS_DOWNLOAD_ID',  # replace with actual GDrive ID
        'data': 'Gaze360',
        'fold': 'all',
        'num_labels': 2,
        'model_name': 'mlp',
        'mlp_input_size': GAZE_FEAT_DIM,
        'dropout': 0.1,
        'hidden_dim': 128,
        'half_precision': False,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'device': 'cpu',
    })

    set_seed(opts.seed)
    opts.device = device

    solver = solver_gaze_image(opts).to(device)
    preds = solver.run_video(aligned_frames_path_list)
    return format_output_video(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to an aligned face image.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--weights_download_dir", type=str, default="./weights_libreface")

    args = parser.parse_args()

    result = estimate_gaze(image_path=args.image_path,
                           device=args.device,
                           weights_download_dir=args.weights_download_dir)
    print(f"Estimated gaze - {result}")


if __name__ == "__main__":
    main()