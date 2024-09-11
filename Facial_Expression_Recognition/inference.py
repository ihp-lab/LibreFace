import os
import argparse


from utils import set_seed, get_data_loaders 
from solver_inference import solver_inference

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--train_csv', type=str, default='training_filtered.csv')
parser.add_argument('--test_csv', type=str, default='validation_filtered.csv')
# storage
parser.add_argument('--data_root', type=str, default='/home/ICT2000/dchang/AffectNet/data/')
parser.add_argument('--ckpt_path', type=str, default='./checkpoints_fm_resnet')
# data
parser.add_argument('--data', type=str, default='AffectNet')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--num_labels', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--sigma', type=float, default=10.0)

# model
parser.add_argument('--student_model_name', type=str, default='resnet', choices=['resnet_heatmap','resnet','swin','mae','emotionnet_mae','gh_feat'])

#distillation
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--T', type=float, default=1.0)
parser.add_argument('--fm_distillation', default=True)
parser.add_argument('--grad',  default=True)
# training
parser.add_argument('--interval', type=int, default=500)
parser.add_argument('--threshold', type=float, default=0)
parser.add_argument('--loss', type=str, default='unweighted')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=3e-5)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--clip', type=int, default=1.0)
parser.add_argument('--when', type=int, default=10, help='when to decay learning rate')
parser.add_argument('--patience', type=int, default=10, help='early stopping')
# device
parser.add_argument('--device', type=str, default='cuda', choices=['cpu','cuda'])

opts = parser.parse_args()
print(opts)
os.makedirs(opts.ckpt_path,exist_ok=True)

# Fix random seed
set_seed(opts.seed)

train_loader, test_loader = get_data_loaders(opts)


# Setup solver 
solver = solver_inference(opts).cuda()
# Start training
solver.run(train_loader, test_loader)