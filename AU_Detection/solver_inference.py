import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn

from utils import get_data_loader
from data_utils import heatmap2au

from models.resnet18 import ResNet18

import time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class solver_in_domain(nn.Module):
	def __init__(self, config):
		super(solver_in_domain, self).__init__()
		self.config = config

		# Setup number of labels
		self.config.num_labels = 12
		self.num_labels = self.config.num_labels

		# Initiate data loaders
		self.get_data_loaders()

		# Initiate the networks
		if config.model_name == "resnet":
			self.model = ResNet18(config).cuda()
		else:
			raise NotImplementedError

   
		if self.config.half_precision:
			print("Use Half Precision.")

		self.criterion = nn.BCEWithLogitsLoss()
		
		print("Number of params: ",count_parameters(self.model))
		# Setup AU index
		if self.config.data == 'BP4D':
			self.aus = [1,2,4,6,7,10,12,14,15,17,23,24]
		# Select the best ckpt
		self.best_val_metric = 0.


	def get_data_loaders(self):
		data = self.config.data
		data_root = self.config.data_root

		fold = self.config.fold
		test_csv = os.path.join(data_root, data, 'labels', fold, 'test.csv')

		self.test_loader = get_data_loader(test_csv, False, self.config)


	def test_model(self, test_loader):
		with torch.no_grad():
			self.eval()
			total_loss, total_sample = 0., 0
			pred_list, gt_list, f1_list = [], [], []
			total_sample = 0
			start_time = time.time()
			for (images, labels) in tqdm(test_loader):
				images, labels = images.cuda(), labels.cuda()
				batch_size = images.shape[0]
				if self.config.half_precision:
					
					images, labels = images.half(), labels.half()
					self.model = self.model.half()

				labels_pred = self.model(images)
				if self.config.half_precision:
					labels_pred = labels_pred.float()

				loss = self.criterion(labels_pred.reshape(-1), labels.reshape(-1))
				total_loss += loss.item()*batch_size
				total_sample += batch_size
				labels_pred = (labels_pred >= 0.5).int()
				pred_list.append(labels_pred)
				gt_list.append(labels)
			time_used = time.time() - start_time
			print("time per sample:",time_used / total_sample)
   			
			avg_loss = total_loss / total_sample
			pred_list = torch.cat(pred_list, dim=0).detach().cpu().numpy()
			gt_list = torch.cat(gt_list, dim=0).detach().cpu().numpy()
			
			for i in range(self.num_labels):
				f1_list.append(100.0*f1_score(gt_list[:, i], pred_list[:, i]))
	
			return avg_loss, f1_list


	def print_metric(self, f1_list, prefix):
		print('{} avg F1: {:.2f}'.format(prefix, sum(f1_list)/len(f1_list)))
		for i in range(len(self.aus)):
			print('AU {}: {:.2f}'.format(self.aus[i], f1_list[i]), end=' ')
		print('')


	def load_best_ckpt(self):
		ckpt_name = os.path.join(self.config.ckpt_path, self.config.data, self.config.fold, self.config.model_name+'.pt')
		checkpoints = torch.load(ckpt_name)['model']
		self.model.load_state_dict(checkpoints, strict=True)



	def run(self):
		best_val_f1 = 0.

		patience = self.config.patience
		torch.backends.cudnn.benchmark = True

		# Test model
		self.load_best_ckpt()
		_, test_f1 = self.test_model(self.test_loader)
		self.print_metric(test_f1, 'Test')
