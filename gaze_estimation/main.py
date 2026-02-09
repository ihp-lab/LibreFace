import os
import argparse

from solver_in_domain import solver_in_domain
from utils import set_seed
from solver_gh import solver_gh
from solver_lm import solver_lm


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)

# storage
parser.add_argument('--data_root', type=str, default='/data/perception-working/aehsieh/gaze-estimation/data')
parser.add_argument('--ckpt_path', type=str, default='/data/perception-working/aehsieh/gaze-estimation/checkpoints')

# data
parser.add_argument('--data', type=str, default='Gaze360', choices=['Gaze360', 'MPII'])
parser.add_argument('--fold', type=str, default='0', choices=['0', '1', '2', '3', '4','all'])
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--num_labels', type=int, default=2)
parser.add_argument('--sigma', type=float, default=10.0)

# model
parser.add_argument('--model_name', type=str, default='mlp', choices=['gaze_mae', 'mlp'])
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--ffhq_pretrain', type=str, default=None)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--pcc_loss', action='store_true')
parser.add_argument('--fm_distillation',  action='store_true')
parser.add_argument('--lm',  action='store_true')

# training
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--interval', type=int, default=500)
parser.add_argument('--threshold', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=0.0003)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--loss', type=str, default='unweighted')
parser.add_argument('--clip', type=int, default=1.0)
parser.add_argument('--when', type=int, default=10, help='when to decay learning rate')
parser.add_argument('--patience', type=int, default=100, help='early stopping')
parser.add_argument('--log_dir',type = str, default = 'tb_runs')



# device
parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'])

opts = parser.parse_args()
print(opts)
os.makedirs(opts.ckpt_path,exist_ok=True)

# Fix random seed
set_seed(opts.seed)

# Setup solver 
if opts.model_name == 'gh_feat':
    solver = solver_gh(opts).cuda()
else:
    if opts.lm:
        solver = solver_lm(opts).cuda()
    else:
        solver = solver_in_domain(opts).cuda()


# Start training
solver.run()