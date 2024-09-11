import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn

from utils import get_data_loader
from models.resnet18 import ResNet18
from models.mae import MaskedAutoEncoder
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class solver_fm_distillation_grad(nn.Module):
	def __init__(self, config):
		super(solver_fm_distillation_grad, self).__init__()
		self.config = config

		self.num_labels = self.config.num_labels

		# Initiate data loaders
		self.get_data_loaders()

		# Initiate the networks
		if config.student_model_name == "resnet":
			self.student_model = ResNet18(config).cuda()
			if config.student_model_path is not None:
				print("Load pretrain weights from FFHQ/AffectNet ...")
				checkpoints = torch.load(config.student_model_path)['model']
				del checkpoints['classifier.4.weight']
				del checkpoints['classifier.4.bias']
				self.student_model.load_state_dict(checkpoints, strict=False)
		else:
			raise NotImplementedError

		if config.teacher_model_name == "emotionnet_mae":
			self.teacher_model = MaskedAutoEncoder(config).cuda()
			if config.teacher_model_path is not None:
				teacher_model_path = os.path.join(self.config.teacher_model_path, self.config.data, self.config.fold, self.config.teacher_model_name+'.pt')
				print("Load pretrain weights from DISFA, path : ",teacher_model_path)
				checkpoints = torch.load(teacher_model_path)['model']
				self.teacher_model.load_state_dict(checkpoints, strict=True)
		else:
			raise NotImplementedError

		# Setup the optimizers and loss function
		opt_params = list(self.student_model.parameters())
		self.optimizer = torch.optim.AdamW(opt_params, lr=config.learning_rate, weight_decay=config.weight_decay)
		self.criterion = nn.MSELoss()
		
		print("Number of params: ",count_parameters(self.student_model))
		# Setup AU index
		self.aus = [1,2,4,5,6,9,12,15,17,20,25,26]

		# Select the best ckpt
		self.best_val_metric = -1.0

	def loss_fn_kd(self, outputs, teacher_outputs):
		"""
		Compute the knowledge-distillation (KD) loss given outputs, labels.
		"Hyperparameters": temperature and alpha
		NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
		and student expects the input tensor to be log probabilities! See Issue #2
		"""
		kldiv_loss = nn.KLDivLoss(reduction="batchmean")
		KD_loss = kldiv_loss(torch.log(F.relu(outputs)+1e-7),
								F.relu(teacher_outputs)+1e-7)
		return KD_loss

	def get_data_loaders(self):
		data = self.config.data
		data_root = self.config.data_root

		fold = self.config.fold
		train_csv = os.path.join(data_root, data, 'labels_intensity_5', fold, 'train.csv')
		val_csv = os.path.join(data_root, data, 'labels_intensity_5', fold, 'test.csv')
		test_csv = os.path.join(data_root, data, 'labels_intensity_5', fold, 'test.csv')

		self.train_loader = get_data_loader(train_csv, True, self.config)
		self.val_loader = get_data_loader(val_csv, False, self.config)
		self.test_loader = get_data_loader(test_csv, False, self.config)


  
	def train_model(self, train_loader):
		self.student_model.train()
		self.teacher_model.eval()
		total_loss, total_sample = 0., 0
		
		for (images, labels, heatmaps) in tqdm(train_loader):
			images, heatmaps, labels = images.cuda(), heatmaps.cuda(), labels.cuda()
			
			batch_size = images.shape[0]
			with torch.no_grad():
				teacher_pred, teacher_feature = self.teacher_model(images)



			self.optimizer.zero_grad()
			student_pred, student_feature = self.student_model(images)
			# Prediction from student model
			student_pred = student_pred * 5.0
			
   			# Align the shape of features 
			student_feature = torch.nn.functional.interpolate(student_feature.unsqueeze(1),size=[self.config.hidden_dim]).squeeze(1)

			student_teacher_pred = self.teacher_model.interpreter(student_feature)

			l2loss = torch.nn.MSELoss()
   
			# feature matching loss for feature-wise distillation
			fm_loss = l2loss(student_feature,teacher_feature)

			# KL Loss 
			kl_loss = self.loss_fn_kd(student_teacher_pred, teacher_pred)
   
			# Overall loss
			loss = fm_loss * self.config.alpha + kl_loss * self.config.alpha + self.criterion(student_pred.reshape(-1), labels.reshape(-1))  
			loss.backward()

			torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.config.clip)
			self.optimizer.step()

			total_loss += loss.item()*batch_size
			total_sample += batch_size

		avg_loss = total_loss / total_sample

		return avg_loss


	def val_model(self, val_loader):
		val_mse, val_mae, val_pcc = self.test_model(val_loader)
		self.save_best_ckpt(sum(val_pcc)/len(val_pcc))

		return val_mse, val_mae, val_pcc


	def test_model(self, test_loader):
		with torch.no_grad():
			self.student_model.eval()
			pred_list, gt_list = [], []
			mse_list, mae_list, pcc_list = [], [], []

			for (images, labels) in tqdm(test_loader):
				images, labels = images.cuda(), labels.cuda()

				self.optimizer.zero_grad()
				labels_pred, _ = self.student_model(images)
				labels_pred = labels_pred * 5.0


				pred_list.append(labels_pred)
				gt_list.append(labels)

			pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()
			gt_list = torch.cat(gt_list, dim=0).detach().cpu().numpy()

			for i in range(self.num_labels):
				mse_list.append(mean_squared_error(gt_list[:, i], pred_list[:, i]))
				mae_list.append(mean_absolute_error(gt_list[:, i], pred_list[:, i]))
				pcc = np.ma.corrcoef(pred_list[:, i], gt_list[:, i])[0][1]
				pcc_list.append(pcc)

			return mse_list, mae_list, pcc_list


	def print_metric(self, mse_list, mae_list, pcc_list, prefix):
		print('{} avg MSE: {:.2f} avg MAE: {:.2f} avg PCC: {:.2f}'.format(prefix, sum(mse_list)/len(mse_list), sum(mae_list)/len(mae_list), sum(pcc_list)/len(pcc_list)))
		print('MSE')
		for i in range(len(self.aus)):
			print('AU {}: {:.2f}'.format(self.aus[i], mse_list[i]), end=' ')
		print('')
		print('MAE')
		for i in range(len(self.aus)):
			print('AU {}: {:.2f}'.format(self.aus[i], mae_list[i]), end=' ')
		print('')
		print('PCC')
		for i in range(len(self.aus)):
			print('AU {}: {:.2f}'.format(self.aus[i], pcc_list[i]), end=' ')
		print('')


	def load_best_ckpt(self):
		ckpt_name = os.path.join(self.config.ckpt_path, self.config.data, self.config.fold, self.config.student_model_name+'.pt')
		checkpoints = torch.load(ckpt_name)['model']
		self.student_model.load_state_dict(checkpoints, strict=True)


	def save_best_ckpt(self, val_metric):
		def update_metric(val_metric):
			if val_metric > self.best_val_metric:
				self.best_val_metric = val_metric
				return True
			return False

		if update_metric(val_metric):
			os.makedirs(os.path.join(self.config.ckpt_path, self.config.data, self.config.fold), exist_ok=True)
			ckpt_name = os.path.join(self.config.ckpt_path, self.config.data, self.config.fold, self.config.student_model_name+'.pt')
			torch.save({'model': self.student_model.state_dict()}, ckpt_name)
			print('save to:', ckpt_name)


	def run(self):
		best_val_pcc = -1.0

		patience = self.config.patience
		for epochs in range(1, self.config.num_epochs+1):
			print('Epoch: {}/{}'.format(epochs, self.config.num_epochs))

			# Train model
			train_loss = self.train_model(self.train_loader)
			print('Training loss: {:.6f}'.format(train_loss))

			# Validate model
			val_mse, val_mae, val_pcc = self.val_model(self.val_loader)
			self.print_metric(val_mse, val_mae, val_pcc, 'Val')
   
			if sum(val_pcc)/len(val_pcc) > best_val_pcc:
				patience = self.config.patience
				best_val_pcc = sum(val_pcc)/len(val_pcc)
			else:
				patience -= 1
				if patience == 0:
					break

		# Test model
		self.load_best_ckpt()
		test_mse, test_mae, test_pcc = self.test_model(self.test_loader)
		self.print_metric(test_mse, test_mae, test_pcc, 'Test')
