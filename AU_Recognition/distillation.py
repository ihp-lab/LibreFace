import os
import argparse
from utils import set_seed
from solver_fm_distillation_grad import solver_fm_distillation_grad

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)

# storage
parser.add_argument('--data_root', type=str, default='/home/ICT2000/dchang/TAC_project/Face_Heatmap/data')
parser.add_argument('--ckpt_path', type=str, default='./resnet_disfa_all')

# data
parser.add_argument('--data', type=str, default='DISFA', choices=['BP4D', 'DISFA'])
parser.add_argument('--fold', type=str, default='all', choices=['0', '1', '2', '3', '4','all'])
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--num_labels', type=int, default=12)
parser.add_argument('--sigma', type=float, default=10.0)

# model
parser.add_argument('--teacher_model_name', type=str, default='emotionnet_mae', choices=['resnet_heatmap','resnet','swin','mae','emotionnet_mae','gh_feat'])
parser.add_argument('--teacher_model_path', type=str, default='/home/ICT2000/dchang/TAC_project/Face_Heatmap/checkpoints_ffhq_mae/')
parser.add_argument('--student_model_name', type=str, default='resnet', choices=['resnet_heatmap','resnet','swin','mae','emotionnet_mae','gh_feat'])
parser.add_argument('--student_model_path', type=str, default=None)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--hidden_dim', type=int, default=128)

#distillation
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--fm_distillation', default=True)



# training
parser.add_argument('--num_epochs', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--clip', type=int, default=1.0)
parser.add_argument('--when', type=int, default=10, help='when to decay learning rate')
parser.add_argument('--patience', type=int, default=5, help='early stopping')

# device
parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'])

opts = parser.parse_args()
print(opts)
os.makedirs(opts.ckpt_path,exist_ok=True)

# Fix random seed
set_seed(opts.seed)

# Setup solver 
solver = solver_fm_distillation_grad(opts).cuda()

# Start training
solver.run()
