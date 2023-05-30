import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error,f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn


from models.resnet18 import ResNet
from models.mae import MaskedAutoEncoder
import torch.nn.functional as F
import pdb

class solver_fm_distillation_grad(nn.Module):
	def __init__(self, config):
		super(solver_fm_distillation_grad, self).__init__()
		self.config = config


		# Initiate the networks
		if config.student_model_name == "resnet":
			self.student_model = ResNet(config).cuda()
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
				teacher_model_path = os.path.join(self.config.teacher_model_path, self.config.data, self.config.teacher_model_name+'.pt')
				print("Load pretrain weights from AffectNet, path : ",teacher_model_path)
				checkpoints = torch.load(teacher_model_path)['model']
				self.teacher_model.load_state_dict(checkpoints, strict=True)
		else:
			raise NotImplementedError

		# Setup the optimizers and loss function
		opt_params = list(self.student_model.parameters())
		self.optimizer = torch.optim.AdamW(opt_params, lr=config.learning_rate, weight_decay=config.weight_decay)
		self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=config.when, factor=0.5, verbose=False)
		self.criterion = nn.CrossEntropyLoss(reduction="mean")
		self.best_val_metric = 0.
  
	def loss_fn_kd(self, outputs, teacher_outputs, T):
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



  
	def train_model(self, train_loader):
		self.student_model.train()
		self.teacher_model.eval()
		total_loss = 0.
		total_acc = 0.
		total_sample = 0
		for (images, labels) in tqdm(train_loader):
			images, labels = images.cuda(), labels.cuda()
			batch_size = images.shape[0]
			with torch.no_grad():
				teacher_pred, teacher_feature = self.teacher_model(images)



			self.optimizer.zero_grad()
			student_pred, student_feature = self.student_model(images)

			student_feature = torch.nn.functional.interpolate(student_feature.unsqueeze(1),size=[self.config.hidden_dim]).squeeze(1)
			student_teacher_pred = self.teacher_model.interpreter(student_feature)
			l2loss = torch.nn.MSELoss()
			fm_loss = l2loss(student_feature,teacher_feature)
			kl_loss = self.loss_fn_kd(student_teacher_pred, teacher_pred, self.config.T)
			loss = fm_loss * self.config.alpha + kl_loss * self.config.alpha + self.criterion(student_pred, labels)  
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.config.clip)
			self.optimizer.step()
			student_pred = torch.argmax(student_pred, 1)
			total_loss += loss.item()*batch_size
			acc = (100.0*torch.sum(student_pred==labels)) / batch_size
			total_acc += acc.item() * batch_size
			total_sample += batch_size

		avg_loss = total_loss / total_sample
		avg_acc = total_acc / total_sample

		return avg_loss, avg_acc

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

	def run(self, train_loader, test_loader):
		best_val_acc = 0.
		best_epoch = 0
		patience = self.config.patience
		for epochs in range(1, self.config.num_epochs+1):
			print('Epoch: %d/%d' % (epochs, self.config.num_epochs))

			# Train model
			train_loss, train_acc = self.train_model(train_loader)
			print(' ** Train loss {:.4f} acc {:.2f} **'.format(train_loss, train_acc))
			# Validate model
			val_loss, val_acc = self.test_model(test_loader)
			print(' ** Val loss {:.4f} acc {:.2f} **'.format(val_loss, val_acc))
			if val_acc > best_val_acc:
				patience = self.config.patience
				best_val_acc = val_acc
				best_epoch = epochs
				os.makedirs(os.path.join(self.config.ckpt_path, self.config.data), exist_ok=True)
				ckpt_name = os.path.join(self.config.ckpt_path, self.config.data, self.config.student_model_name+'.pt')
				torch.save({'model': self.student_model.state_dict()}, ckpt_name)
			else:
				patience -= 1
				if patience == 0:
					break
		print('Best test acc {:.2f} from epoch {}'.format(best_val_acc, best_epoch))

