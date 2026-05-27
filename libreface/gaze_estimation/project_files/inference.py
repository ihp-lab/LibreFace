import argparse
import numpy as np
import random
import torch

from libreface.gaze_estimation import gaze_inference


class ConfigObject:
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def set_seed(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def estimate_gaze(
    csv_input: str,
    device: str = "cpu",
    ckpt_path: str | None = None,
    weights_download_dir: str = "./weights_libreface",
    infer_batch_size: int = 64,
    mlp_input_size: int | None = None,
    strict: bool = False,
    save_metrics: bool = False,
    output: str = "gaze_preds.npy",
    model_name: str = "mlp",
):
    """
    Public API for gaze estimation using the original CSV-driven workflow.

    Args:
        csv_input: Path to CSV containing columns:
            - image_feat
            - yaw
            - pitch
        device: "cpu" or "cuda"
        ckpt_path: Path to checkpoint; if None, fall back to default weights path
        infer_batch_size: inference batch size
        mlp_input_size: feature dimension for MLP; if None, infer from first .npy
        strict: whether to strictly match checkpoint keys
        save_metrics: if True, compute MSE/MAE using yaw/pitch in CSV
        output: path to save .npy predictions
        model_name: "mlp" or "gaze_mae"

    Returns:
        np.ndarray of shape (N, 2), typically [yaw, pitch]
    """
    opts = ConfigObject({
        "seed": 42,
        "data_root": "",
        "ckpt_path": "",
        "data": "Gaze360",
        "fold": "0",
        "num_labels": 2,
        "model_name": model_name,
        "dropout": 0.1,
        "hidden_dim": 128,
        "device": device,
        "ckpt": ckpt_path or f"{weights_download_dir}/gaze_estimation/weights/mlp.pt",
        "input_type": "npy",           # inferred from CSV image_feat paths
        "output": output,
        "infer_batch_size": infer_batch_size,
        "mlp_input_size": mlp_input_size,
        "strict": strict,
        "csv_input": csv_input,
        "save_metrics": save_metrics,
    })

    set_seed(opts.seed)
    preds = gaze_inference.run(opts)
    return preds


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--seed", type=int, default=42)

    # kept for style compatibility with original project
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--ckpt_path", type=str, default="")

    p.add_argument("--data", type=str, default="Gaze360", choices=["Gaze360", "MPII"])
    p.add_argument("--fold", type=str, default="0", choices=["0", "1", "2", "3", "4", "all"])
    p.add_argument("--num_labels", type=int, default=2)

    p.add_argument("--model_name", type=str, default="mlp", choices=["gaze_mae", "mlp"])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden_dim", type=int, default=128)

    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    p.add_argument("--ckpt", type=str, required=True)

    p.add_argument("--csv_input", type=str, required=True)

    p.add_argument("--output", type=str, default="inference_result.npy")
    p.add_argument("--infer_batch_size", type=int, default=64)
    p.add_argument(
        "--mlp_input_size",
        type=int,
        default=None,
        help="For MLP: feature dimension. If None, infer from first .npy in CSV.",
    )
    p.add_argument("--strict", action="store_true")
    p.add_argument("--save_metrics", action="store_true")

    args = p.parse_args()

    set_seed(args.seed)
    gaze_inference.run(args)


if __name__ == "__main__":
    main()