import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn


from models.resnet18 import ResNet
import torch.nn.functional as F
import pdb

class solver_inference(nn.Module):
	def __init__(self, config):
		super(solver_inference, self).__init__()
		self.config = config
		# Initiate the networks
		if config.student_model_name == "resnet":
			self.student_model = ResNet(config).cuda()
		else:
			raise NotImplementedError

		# Setup loss function
		self.criterion = nn.CrossEntropyLoss(reduction="mean")
		self.best_val_metric = 0.
  

	def test_model(self, test_loader):
		with torch.no_grad():
			self.student_model.eval()
			total_loss = 0.
			total_acc = 0.
			total_sample = 0
			for (images, labels) in tqdm(test_loader):
				images, labels = images.cuda(), labels.cuda()
				batch_size = images.shape[0]

				labels_pred, _ = self.student_model(images)
				loss = self.criterion(labels_pred, labels)
    
				labels_pred = torch.argmax(labels_pred, 1)

				total_loss += loss.item()*batch_size
				acc = (100.0*torch.sum(labels_pred==labels)) / batch_size
				total_acc += acc.item() * batch_size
				total_sample += batch_size
			avg_loss = total_loss / total_sample
			avg_acc = total_acc / total_sample

			return avg_loss, avg_acc


	def load_best_ckpt(self):
		ckpt_name = os.path.join(self.config.ckpt_path, self.config.data, self.config.student_model_name+'.pt')
		print("Loading weights from: ",ckpt_name)
		checkpoints = torch.load(ckpt_name)['model']
		self.student_model.load_state_dict(checkpoints, strict=True)
  
  
	def run(self, train_loader, test_loader):
		best_val_acc = 0.
		best_epoch = 0
		patience = self.config.patience
		self.load_best_ckpt()
		# Test model
		test_loss, test_acc = self.test_model(test_loader)
		print(' ** Test loss {:.4f} acc {:.2f} **'.format(test_loss, test_acc))

