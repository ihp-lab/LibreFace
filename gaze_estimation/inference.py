import os
import argparse

from utils import set_seed
import gaze_inference as gaze_inference

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--data_root', type=str, default='/data/perception-working/aehsieh/gaze-estimation/data')
parser.add_argument('--ckpt_path', type=str, default='/data/perception-working/aehsieh/gaze-estimation/checkpoints')

# data/model (keep same style, minimal fields needed)
parser.add_argument('--data', type=str, default='Gaze360', choices=['Gaze360', 'MPII'])
parser.add_argument('--fold', type=str, default='0', choices=['0', '1', '2', '3', '4','all'])
parser.add_argument('--num_labels', type=int, default=2)

parser.add_argument('--model_name', type=str, default='mlp', choices=['gaze_mae', 'mlp'])
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--hidden_dim', type=int, default=128)

parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'])

# parser.add_argument('--ckpt', type=str, default="/data/perception-working/aehsieh/gaze-estimation/checkpoints/Gaze360/mlp/mlp_256/mlp_unfiltered.pt")
parser.add_argument('--ckpt', type=str, default="/home/xguan/libreface2_cleancode/LibreFace/gaze_estimation/checkpoints_mlp/mlp_unfiltered.pt")

parser.add_argument('--input_type', type=str, default='npy', choices=['npy', 'image'])
parser.add_argument('--output', type=str, default='inference_result.npy')
parser.add_argument('--infer_batch_size', type=int, default=64)
parser.add_argument('--mlp_input_size', type=int, default=None,
                    help='For MLP: feature dimension. If None and input_type=npy, will infer from first file.')
parser.add_argument('--strict', action='store_true')
parser.add_argument('--csv', type=str, default=None)
parser.add_argument('--save_metrics', default = "true")

opts = parser.parse_args()
print(opts)

os.makedirs(opts.ckpt_path, exist_ok=True)

set_seed(opts.seed)

gaze_inference.run(opts)

# python inference.py  --csv /data/perception-working/aehsieh/gaze-estimation/labels/all_features/test_all_feat_30.csv  --output results/preds_test.npy  