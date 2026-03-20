
import os
import glob
import numpy as np
import torch

from models.mae import MaskedAutoEncoder
from models.mlp import MLP

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm


def build_model(opts):
    
    if opts.model_name == "gaze_mae":
        return MaskedAutoEncoder(opts)
    elif opts.model_name == "mlp":
        if opts.mlp_input_size is None:
            raise ValueError(
                "opts.mlp_input_size is None for MLP. "
                "Set --mlp_input_size or use input_type=npy so it can be inferred."
            )
        return MLP(opts.mlp_input_size, opts.num_labels, 256)
    else:
        raise NotImplementedError(f"Unknown model_name: {opts.model_name}")





def load_checkpoint(model, ckpt_path, device, strict=True):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    

    model.load_state_dict(ckpt, strict=strict)


def collect_inputs(input_path, input_type):
    """
    input_path can be a file or a folder.
    """
    if input_type == "npy":
        if os.path.isdir(input_path):
            paths = sorted(glob.glob(os.path.join(input_path, "*.npy")))
            if not paths:
                raise FileNotFoundError(f"No .npy files found in folder: {input_path}")
            return paths
        else:
            if not os.path.exists(input_path):
                raise FileNotFoundError(input_path)
            if not input_path.endswith(".npy"):
                raise ValueError(f"--input_type npy requires a .npy file, got: {input_path}")
            return [input_path]

    if input_type == "image":
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        if os.path.isdir(input_path):
            paths = []
            for e in exts:
                paths.extend(glob.glob(os.path.join(input_path, e)))
            paths = sorted(paths)
            if not paths:
                raise FileNotFoundError(f"No images found in folder: {input_path}")
            return paths
        else:
            if not os.path.exists(input_path):
                raise FileNotFoundError(input_path)
            return [input_path]

    raise ValueError(f"Unknown input_type: {input_type}")

def collect_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    required = ["image_feat", "yaw", "pitch"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns {missing}. Found: {list(df.columns)}")

    inputs = df["image_feat"].astype(str).tolist()   
    labels = df[["yaw", "pitch"]].to_numpy(dtype=np.float32)
    return inputs, labels


def npy_to_batch(npy_paths):
    
    feats = []
    for p in npy_paths:
        x = np.load(p)
        x = np.asarray(x).reshape(-1) 
        feats.append(x)
    return np.stack(feats, axis=0)  


def images_to_batch_landmarks(image_paths, opts):
    
    raise NotImplementedError(
        "image->landmarks extraction is not implemented. "
        "Use --input_type npy, or implement images_to_batch_landmarks() to match training."
    )


@torch.no_grad()
def run_inference(model, inputs, opts, device):
    model.eval()

    preds_all = []
    batch_size = int(getattr(opts, "infer_batch_size", 64))

    total = len(inputs)
    num_batches = (total + batch_size - 1) // batch_size

    for i in tqdm(range(0, total, batch_size),total=num_batches,desc="Running inference",unit="batch"):
        batch_inputs = inputs[i:i + batch_size]

        if opts.input_type == "npy":
            feats = npy_to_batch(batch_inputs)  # (B, D)
            feats = torch.from_numpy(feats).float().to(device)
        else:
            feats = images_to_batch_landmarks(batch_inputs, opts).to(device)

        preds = model(feats)  # (B, num_labels)
        preds_all.append(preds.detach().cpu().numpy())

    return np.concatenate(preds_all, axis=0)


def save_outputs(inputs, preds, output_path):
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    np.save(output_path, preds)

    # sidecar txt
    if output_path.endswith(".npy"):
        txt_out = output_path[:-4] + ".txt"
    else:
        txt_out = output_path + ".txt"

    with open(txt_out, "w") as f:
        for p, y in zip(inputs, preds):
            f.write(f"{p}\t" + "\t".join([f"{v:.6f}" for v in y.tolist()]) + "\n")

    return txt_out


def infer_mlp_input_size_if_needed(opts, inputs):
    
    if opts.model_name != "mlp":
        return

    if opts.mlp_input_size is not None:
        return

    if opts.input_type != "npy":
        raise ValueError(
            "For MLP with input_type=image, you must set --mlp_input_size "
            "and implement image->landmarks extraction."
        )

    first_feat = np.load(inputs[0]).reshape(-1)
    opts.mlp_input_size = int(first_feat.shape[0])
    print(f"[INFO] Inferred mlp_input_size={opts.mlp_input_size} from: {inputs[0]}")


def get_device(opts):
    if opts.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run(opts):
    device = get_device(opts)
    if getattr(opts, "csv_input", None):
        inputs, labels = collect_from_csv(opts.csv_input)
        opts.input_type = "npy"
    else:
        labels = None
        inputs = collect_inputs(opts.input, opts.input_type)

    ckpt_path = opts.ckpt
    # print("[DEBUG] opts.ckpt =", getattr(opts, "ckpt", None))
    # print("[DEBUG] opts.ckpt_path =", getattr(opts, "ckpt_path", None))
    # print("[DEBUG] opts.data/fold/model_name =", getattr(opts, "data", None), getattr(opts, "fold", None), getattr(opts, "model_name", None))
    if ckpt_path is None:
        raise ValueError("opts.ckpt is None. Please specify --ckpt.")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    infer_mlp_input_size_if_needed(opts, inputs)
    model = build_model(opts).to(device)

    strict = bool(getattr(opts, "strict", False))
   
    load_checkpoint(model, ckpt_path, device, strict=strict)

    preds = run_inference(model, inputs, opts, device)
    if labels is not None and getattr(opts, "save_metrics", False):
        mse = [mean_squared_error(labels[:, i], preds[:, i]) for i in range(preds.shape[1])]
        mae = [mean_absolute_error(labels[:, i], preds[:, i]) for i in range(preds.shape[1])]
        print(f"[METRIC] MSE yaw={mse[0]:.6f} pitch={mse[1]:.6f} avg={(mse[0]+mse[1])/2:.6f}")
        print(f"[METRIC] MAE yaw={mae[0]:.6f} pitch={mae[1]:.6f} avg={(mae[0]+mae[1])/2:.6f}")
    txt_out = save_outputs(inputs, preds, opts.output)

    print(f"[OK] Loaded ckpt: {ckpt_path}")
    print(f"[OK] Saved predictions: {opts.output}")
    print(f"[OK] Saved mapping txt: {txt_out}")

    for p, y in list(zip(inputs, preds))[:5]:
        print("[PRED]", p, y)

    return preds