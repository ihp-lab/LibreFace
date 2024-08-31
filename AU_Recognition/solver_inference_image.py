import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torchvision import transforms

from utils import get_data_loader
from models.resnet18 import ResNet18
from models.mae import MaskedAutoEncoder
import time

import matplotlib.pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class image_test(object):
	def __init__(self, img_size=256, crop_size=224):
		self.img_size = img_size
		self.crop_size = crop_size

	def __call__(self, img):
		transform = transforms.Compose([
			transforms.Resize(self.img_size),
			transforms.CenterCrop(self.crop_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])
		])
		img = transform(img)

		return img

class solver_inference_image(nn.Module):
	def __init__(self, config):
		super(solver_inference_image, self).__init__()
		self.config = config

		# Setup number of labels

		self.config.num_labels = 12
		self.num_labels = self.config.num_labels

		# Initiate data loaders
		# self.get_data_loaders()
		self.image_transform = image_test(img_size=config.image_size, crop_size=config.crop_size)

		# Initiate the networks
		if config.model_name == "resnet":
			self.model = ResNet18(config).cuda()
		elif config.model_name == "emotionnet_mae":
			self.model = MaskedAutoEncoder(config).cuda()
		else:
			raise NotImplementedError

   
		if self.config.half_precision:
			print("Use Half Precision.")
   
		# Setup the optimizers and loss function
		opt_params = list(self.model.parameters())
		self.optimizer = torch.optim.AdamW(opt_params, lr=config.learning_rate, weight_decay=config.weight_decay)
		self.criterion = nn.MSELoss()
		print("Number of params: ",count_parameters(self.model))
		# Setup AU index
		self.aus = [1,2,4,5,6,9,12,15,17,20,25,26]

		# Select the best ckpt
		self.best_val_metric = -1.0


	# def get_data_loaders(self):
	# 	data = self.config.data
	# 	data_root = self.config.data_root
	# 	fold = self.config.fold
	# 	test_csv = os.path.join(data_root, data, 'labels_intensity_5', fold, 'test.csv')
	# 	self.test_loader = get_data_loader(test_csv, False, self.config)

	def pil_loader(self, path):
		with open(path, 'rb') as f:
			with Image.open(f) as img:
				return img.convert('RGB')

	def transform_image_inference(self, aligned_image_path):

		image = self.pil_loader(aligned_image_path)
		# print(image.shape)
		image = self.image_transform(image)

		return image


	def image_inference(self, transformed_image):
		with torch.no_grad():
			self.eval()
			input_image = torch.unsqueeze(transformed_image, 0).cuda()
			if self.config.half_precision:
				input_image = input_image.half()
				self.model = self.model.half()
			labels_pred = self.model(input_image)
			if self.config.half_precision:
				labels_pred = labels_pred.float()
			labels_pred = torch.clamp(labels_pred * 5.0, min=0.0, max=5.0)
			return labels_pred


	def test_model(self, test_loader):
		with torch.no_grad():
			self.eval()
			pred_list, gt_list = [], []
			mse_list, mae_list, pcc_list = [], [], []
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
				labels_pred = torch.clamp(labels_pred * 5.0, min=0.0, max=5.0)
				pred_list.append(labels_pred)
				gt_list.append(labels)
    
				total_sample += batch_size
			time_used = time.time() - start_time
			print("time per sample:",time_used / total_sample)
   
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
		print("Loading weights from: ",ckpt_name)
		checkpoints = torch.load(ckpt_name)['model']
		self.model.load_state_dict(checkpoints, strict=True)



	def run(self, aligned_image_path):
		torch.backends.cudnn.benchmark = True

		# Test model
		self.load_best_ckpt()
		transformed_image = self.transform_image_inference(aligned_image_path)
		pred_labels = self.image_inference(transformed_image)
		print(pred_labels)
		print(pred_labels.size())
		# test_mse, test_mae, test_pcc = self.test_model(self.test_loader)

		# self.print_metric(test_mse, test_mae, test_pcc, 'Test')
