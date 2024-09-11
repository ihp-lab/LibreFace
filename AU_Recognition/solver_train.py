import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn

from utils import get_data_loader
from models.resnet18 import ResNet18
from models.mae import MaskedAutoEncoder


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class solver_train(nn.Module):
	def __init__(self, config):
		super(solver_train, self).__init__()
		self.config = config

		# Setup number of labels
		self.num_labels = self.config.num_labels

		# Initiate data loaders
		self.get_data_loaders()

		# Initiate the networks
		if config.model_name == "resnet":
			self.model = ResNet18(config).cuda()
			if config.ffhq_pretrain is not None:
				print("Load resnet18 pretrain weights from FFHQ ...")
				checkpoints = torch.load(config.ffhq_pretrain)['model']
				del checkpoints['classifier.4.weight']
				del checkpoints['classifier.4.bias']
				self.model.load_state_dict(checkpoints, strict=False)
		elif config.model_name == "emotionnet_mae":
			self.model = MaskedAutoEncoder(config).cuda()
			if config.ffhq_pretrain is not None:
				print("Load MaskedAutoEncoder pretrain weights from FFHQ ...")
				checkpoints = torch.load(config.ffhq_pretrain)['model']
				del checkpoints['interpreter.4.weight']
				del checkpoints['interpreter.4.bias']
				self.model.load_state_dict(checkpoints, strict=False)
		else:
			raise NotImplementedError

		# Setup the optimizers and loss function
		opt_params = list(self.model.parameters())
		self.optimizer = torch.optim.AdamW(opt_params, lr=config.learning_rate, weight_decay=config.weight_decay)
		self.criterion = nn.MSELoss()
		
		print("Number of params: ",count_parameters(self.model))
		# Setup AU index
		self.aus = [1,2,4,5,6,9,12,15,17,20,25,26]

		# Select the best ckpt
		self.best_val_metric = -1.0


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
		self.train()
		total_loss, total_sample = 0., 0
		
		for (images, labels, heatmaps) in tqdm(train_loader):
			images, heatmaps, labels = images.cuda(), heatmaps.cuda(), labels.cuda()
			
			batch_size = images.shape[0]

			self.optimizer.zero_grad()
			pred = self.model(images)
			pred = pred * 5.0
			loss = self.criterion(pred.reshape(-1), labels.reshape(-1))
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
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
			self.eval()
			pred_list, gt_list = [], []
			mse_list, mae_list, pcc_list = [], [], []

			for (images, labels) in tqdm(test_loader):
				images, labels = images.cuda(), labels.cuda()

				labels_pred = self.model(images)
				loss = self.criterion(labels_pred.reshape(-1), labels.reshape(-1))
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
			os.makedirs(os.path.join(self.config.ckpt_path, self.config.data, self.config.fold), exist_ok=True)
			ckpt_name = os.path.join(self.config.ckpt_path, self.config.data, self.config.fold, self.config.model_name+'.pt')
			torch.save({'model': self.model.state_dict()}, ckpt_name)
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
