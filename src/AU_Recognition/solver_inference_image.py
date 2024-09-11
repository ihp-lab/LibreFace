import os
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from src.AU_Recognition.models.resnet18 import ResNet18
from src.AU_Recognition.models.mae import MaskedAutoEncoder

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
   
		print("Number of params: ",count_parameters(self.model))
		# Setup AU index
		self.aus = [1,2,4,5,6,9,12,15,17,20,25,26]

		# Select the best ckpt
		self.best_val_metric = -1.0

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
