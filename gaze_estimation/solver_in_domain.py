import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn

from utils import get_data_loader_landmark, angles_to_vector, vector_to_angles, transform_gaze_vector_batch
from data_utils import heatmap2au
from models.mae import MaskedAutoEncoder
from models.mlp import MLP
from datetime import datetime

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Neg_Pearson_Loss(nn.Module):   
    def __init__(self):
        super(Neg_Pearson_Loss,self).__init__()
    def forward(self, X, Y):       
        assert not torch.any(torch.isnan(X))
        assert not torch.any(torch.isnan(Y))
        # Normalise X and Y
        X = X-X.mean(1)[:, None]
        Y = Y- Y.mean(1)[:, None]
        # Standardise X and Y
        X = (X/(X.std(1)+1e-5)[:, None])+1e-5
        Y =(Y/(Y.std(1)+1e-5)[:, None])+1e-5
        #multiply X and Y
        Z=(X*Y).mean(1)
        Z=1-Z.mean()
        return Z

class solver_in_domain(nn.Module):
	def __init__(self, config):
		super(solver_in_domain, self).__init__()
		self.config = config

		# Setup number of labels
		if self.config.data == 'Gaze360':
			self.config.num_labels = 2
		self.num_labels = self.config.num_labels

		# Initiate data loaders
		self.get_data_loaders()

		self.writer = SummaryWriter(log_dir=config.log_dir)

		# Track best validation MAEs
		self.best_val_mae_yaw = float('inf')
		self.best_val_mae_pitch = float('inf')
		self.best_val_mae_avg = float('inf')

		# Initiate the networks
		if config.model_name == 'gaze_mae':
			self.model = MaskedAutoEncoder(config).cuda()
			checkpoint = torch.load('/home/xguan/IHP_projects/rafdb_libre/mae_pretrained_standard/emotionnet_mae.pt')
			print("Load EmotioNet weights from /home/xguan/IHP_projects/rafdb_libre/mae_pretrained_standard/emotionnet_mae.pt")
			state_dict = checkpoint['model']
			state_dict = {k: v for k, v in state_dict.items() if not (k.startswith('interpreter.4.weight') or k.startswith('interpreter.4.bias'))}
			self.model.load_state_dict(state_dict, strict=False)
		elif config.model_name == 'mlp':
			# change sample if using limited vs all features
			sample_feat = np.load('/data/perception-working/aehsieh/gaze-estimation/data/Gaze360_all_features/rec_001/head/000001/001620.npy')
			input_size = sample_feat.size
			self.model = MLP(input_size, config.num_labels, 256).cuda()
			# reduce to 1024, 512, 256
		else:
			raise NotImplementedError

		# Setup the optimizers and loss function
		opt_params = list(self.model.parameters())
		self.optimizer = torch.optim.AdamW(opt_params, lr=config.learning_rate, weight_decay=config.weight_decay)
		if self.config.pcc_loss:
			self.criterion = Neg_Pearson_Loss()
		else:
			self.criterion = nn.MSELoss()
		
		print("Number of params: ", count_parameters(self.model))

		# Select the best ckpt
		self.best_val_metric = -1.0


	def get_data_loaders(self):
		data = self.config.data
		data_root = self.config.data_root

		fold = self.config.fold

		train_csv = os.path.join('/data/perception-working/aehsieh/gaze-estimation/labels/all_features/trainval_all_feat_30.csv')
		val_csv = os.path.join('/data/perception-working/aehsieh/gaze-estimation/labels/all_features/test_all_feat_30.csv')
		test_csv = os.path.join('/data/perception-working/aehsieh/gaze-estimation/labels/all_features/test_all_feat_30.csv')

		self.train_loader = get_data_loader_landmark(train_csv, True, self.config)
		self.val_loader = get_data_loader_landmark(val_csv, False, self.config)
		self.test_loader = get_data_loader_landmark(test_csv, False, self.config)

	def train_model(self, train_loader):
		self.train()
		total_loss, total_sample = 0., 0

		pred_list, gt_list = [], []

		for batch in tqdm(train_loader):
			if batch is None: 
				continue
				
			# images, labels, m_invs = batch
			landmarks, labels = batch
			# m_invs = m_invs.cuda()

			landmarks, labels = landmarks.cuda(), labels.cuda()
			batch_size = landmarks.shape[0]

			self.optimizer.zero_grad()

			# if self.config.model_name == 'resnet_heatmap':
			# 	heatmaps_preds = self.model(images)][[[[[[3]]]]]]
			# 	loss = self.criterion(heatmaps_preds, heatmaps)
			# else:
			
			pred = self.model(landmarks)  # shape: (B, 3) if gaze vector
			# pred = transform_gaze_vector_batch(pred, m_invs)

			loss = self.criterion(pred, labels)

			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
			self.optimizer.step()

			total_loss += loss.item() * batch_size
			total_sample += batch_size

			pred_list.append(pred.detach().cpu())
			gt_list.append(labels.detach().cpu())

		avg_loss = total_loss / total_sample

		pred_all = torch.cat(pred_list, dim=0).numpy()
		gt_all = torch.cat(gt_list, dim=0).numpy()

		train_mse = [mean_squared_error(gt_all[:, i], pred_all[:, i]) for i in range(self.num_labels)]
		train_mae = [mean_absolute_error(gt_all[:, i], pred_all[:, i]) for i in range(self.num_labels)]

		return avg_loss, train_mse, train_mae

	def val_model(self, val_loader):
		val_mse, val_mae, val_pcc = self.test_model(val_loader)
		self.save_best_ckpt(sum(val_pcc)/len(val_pcc))

		return val_mse, val_mae, val_pcc


	def test_model(self, test_loader):
		with torch.no_grad():
			self.eval()
			pred_list, gt_list = [], []
			mse_list, mae_list, pcc_list = [], [], []

			for (landmarks, labels) in tqdm(test_loader):
				# images, labels, m_invs = images.cuda(), labels.cuda(), m_invs.cuda()
				landmarks, labels = landmarks.cuda(), labels.cuda()

				labels_pred = self.model(landmarks)

				# pred_np = labels_pred.detach().cpu().numpy()
				# m_inv_np = m_invs.detach().cpu().numpy()
				# gaze_vecs = angles_to_vector(pred_np[:, 0], pred_np[:, 1])
				# gaze_vecs_orig = np.einsum('nij,nj->ni', m_inv_np, gaze_vecs)
				# pred_np = vector_to_angles(gaze_vecs_orig)

				loss = self.criterion(labels_pred, labels)

				pred_list.append(labels_pred)
				gt_list.append(labels)

				# pred_list.append(torch.from_numpy(pred_np))
				# gt_list.append(labels)

			pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()
			gt_list = torch.cat(gt_list, dim=0).detach().cpu().numpy()

			for i in range(self.num_labels):
				mse_list.append(mean_squared_error(gt_list[:, i], pred_list[:, i]))
				mae_list.append(mean_absolute_error(gt_list[:, i], pred_list[:, i]))
				pcc = np.ma.corrcoef(pred_list[:, i], gt_list[:, i])[0][1]
				pcc_list.append(pcc)

			return mse_list, mae_list, pcc_list


	def print_metric(self, mse_list, mae_list, pcc_list, prefix):
		print('{} avg MSE: {:.2f} avg MAE: {:.2f} avg PCC: {:.2f}'.format(
			prefix,
			sum(mse_list)/len(mse_list),
			sum(mae_list)/len(mae_list),
			sum(pcc_list)/len(pcc_list)
		))

		print('MSE')
		print('Yaw: {:.2f} Pitch: {:.2f}'.format(mse_list[0], mse_list[1]))
		print('MAE')
		print('Yaw: {:.2f} Pitch: {:.2f}'.format(mae_list[0], mae_list[1]))
		print('PCC')
		print('Yaw: {:.2f} Pitch: {:.2f}'.format(pcc_list[0], pcc_list[1]))

	def load_best_ckpt(self):
		ckpt_name = os.path.join(self.config.ckpt_path, self.config.data, self.config.fold, self.config.model_name+'.pt')
		checkpoints = torch.load(ckpt_name)['model']
		self.model.load_state_dict(checkpoints, strict=True)


	def save_best_ckpt(self, val_metric):
		def update_metric(val_metric):
			if val_metric > self.best_val_metric:
				self.best_val_metric = val_metric
				return True
			return False

		if update_metric(val_metric):
			# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			os.makedirs(os.path.join(self.config.ckpt_path, self.config.data, self.config.fold), exist_ok=True)
			ckpt_name = os.path.join(self.config.ckpt_path, self.config.data, self.config.fold, self.config.model_name+'.pt')
			# ckpt_name = os.path.join(self.config.ckpt_path, self.config.data, self.config.fold, 
			# 	f"{self.config.model_name}_{timestamp}.pt"
			# )
			torch.save({'model': self.model.state_dict()}, ckpt_name)
			print('save to:', ckpt_name)


	def run(self):
		best_val_pcc = -1.0

		patience = self.config.patience
		for epochs in range(1, self.config.num_epochs+1):
			
			print('Epoch: {}/{}'.format(epochs, self.config.num_epochs))

			# Train model
			train_loss, train_mse, train_mae = self.train_model(self.train_loader)
			print('Training loss: {:.6f}'.format(train_loss))
			print('Train avg MSE: {:.2f} avg MAE: {:.2f}'.format(
				sum(train_mse)/len(train_mse),
				sum(train_mae)/len(train_mae)
			))
			print('Train MSE: Yaw {:.2f}, Pitch {:.2f}'.format(train_mse[0], train_mse[1]))
			print('Train MAE: Yaw {:.2f}, Pitch {:.2f}'.format(train_mae[0], train_mae[1]))
			
			# Validate model
			val_mse, val_mae, val_pcc = self.val_model(self.val_loader)
			self.print_metric(val_mse, val_mae, val_pcc, 'Val')

			avg_val_mae = sum(val_mae) / len(val_mae)
			if val_mae[0] < self.best_val_mae_yaw:
				self.best_val_mae_yaw = val_mae[0]
			if val_mae[1] < self.best_val_mae_pitch:
				self.best_val_mae_pitch = val_mae[1]
			if avg_val_mae < self.best_val_mae_avg:
				self.best_val_mae_avg = avg_val_mae

			self.writer.add_scalar("Loss/train", train_loss, epochs)
			self.writer.add_scalar("MSE/train/yaw", train_mse[0], epochs)
			self.writer.add_scalar("MSE/train/pitch", train_mse[1], epochs)
			self.writer.add_scalar("MAE/train/yaw", train_mae[0], epochs)
			self.writer.add_scalar("MAE/train/pitch", train_mae[1], epochs)

			self.writer.add_scalar("MSE/val/yaw", val_mse[0], epochs)
			self.writer.add_scalar("MSE/val/pitch", val_mse[1], epochs)
			self.writer.add_scalar("MAE/val/yaw", val_mae[0], epochs)
			self.writer.add_scalar("MAE/val/pitch", val_mae[1], epochs)
			self.writer.add_scalar("PCC/val/yaw", val_pcc[0], epochs)
			self.writer.add_scalar("PCC/val/pitch", val_pcc[1], epochs)
			self.writer.close()
			
			if sum(val_pcc)/len(val_pcc) > best_val_pcc:
				patience = self.config.patience
				best_val_pcc = sum(val_pcc)/len(val_pcc)
			else:
				patience -= 1
				if patience == 0:
					break
		
		print(f"Lowest Val MAE Yaw: {self.best_val_mae_yaw:.2f}")
		print(f"Lowest Val MAE Pitch: {self.best_val_mae_pitch:.2f}")
		print(f"Lowest Val Avg MAE: {self.best_val_mae_avg:.2f}")

		# Test model
		self.load_best_ckpt()
		test_mse, test_mae, test_pcc = self.test_model(self.test_loader)
		self.print_metric(test_mse, test_mae, test_pcc, 'Test')

	# analyze final results
	def test(self):
		self.load_best_ckpt()
		test_mse, test_mae, test_pcc = self.test_model(self.test_loader)
		self.print_metric(test_mse, test_mae, test_pcc, 'Test')