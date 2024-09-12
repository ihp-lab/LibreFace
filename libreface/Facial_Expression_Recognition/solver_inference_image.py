import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms

from libreface.Facial_Expression_Recognition.models.resnet18 import ResNet
from libreface.utils import download_weights

class solver_inference_image(nn.Module):
	def __init__(self, config):
		super(solver_inference_image, self).__init__()
		self.config = config

		self.device = config.device

		# Initiate the networks
		if config.student_model_name == "resnet":
			self.student_model = ResNet(config).to(self.device)
		else:
			raise NotImplementedError

		self.load_best_ckpt()

		self.img_size = (config.image_size, config.image_size)
		self.convert = transforms.ToTensor()
		self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	def read_and_transform_image(self, image_path):
		
		image = Image.open(image_path).resize((self.img_size[0], self.img_size[1]))
		image = self.convert(image)
		image = self.transform(image)
		return image

	def image_inference(self, transformed_image):
		with torch.no_grad():
			self.student_model.eval()
			input_image = torch.unsqueeze(transformed_image, 0).to(self.device)
			labels_pred, _ = self.student_model(input_image)
			labels_pred = torch.argmax(labels_pred, 1)
			return labels_pred

	def load_best_ckpt(self):
		download_weights(self.config.weights_download_id, self.config.ckpt_path)
		checkpoints = torch.load(self.config.ckpt_path, map_location=self.device)['model']
		self.student_model.load_state_dict(checkpoints, strict=True)
  
  
	def run(self, aligned_image_path):
		
		transformed_image = self.read_and_transform_image(aligned_image_path)
		pred_label = self.image_inference(transformed_image)
		pred_label = pred_label.squeeze().tolist()
		return pred_label

