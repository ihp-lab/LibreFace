import torch.nn as nn
import torch
import torchvision
import pdb

class ResNet18(nn.Module):
	def __init__(self, opts):
		super(ResNet18, self).__init__()
		self.fm_distillation = opts.fm_distillation
		self.dropout = opts.dropout
		resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
		resnet18_layers = list(resnet18.children())[:-1]
		self.encoder = nn.Sequential(*resnet18_layers)

		self.classifier = nn.Sequential(
					nn.Linear(512, 128),
					nn.ReLU(),
					nn.BatchNorm1d(num_features=128),
					nn.Dropout(p=self.dropout),
					nn.Linear(128, opts.au_recognition_num_labels),
					nn.Sigmoid()
					) # au recognition
  
		self.classifier_2 = nn.Sequential(
				nn.Linear(512, 128),
				nn.ReLU(),
				nn.BatchNorm1d(num_features=128),
				nn.Dropout(p=self.dropout),
				nn.Linear(128, opts.au_detection_num_labels),
    			nn.Sigmoid()
				) # au detection

		# self.classifier_3 = nn.Sequential(
		# 		nn.Linear(512, 128),
		# 		nn.ReLU(),
		# 		nn.BatchNorm1d(num_features=128),
		# 		nn.Dropout(p=self.dropout),
		# 		nn.Linear(128, opts.fer_num_labels),
		# 		nn.Sigmoid()
   		# 		)		 # facial expression recognition
   
	def forward(self, images, task_name=None):
		batch_size = images.shape[0]
		features = self.encoder(images).reshape(batch_size, -1)
		if task_name==None:
			print("specify task name.")
			raise NotImplementedError
		elif task_name=="au_recognition":
			labels = self.classifier(features)
		elif task_name=="au_detection":
			labels = self.classifier_2(features)
		# elif task_name=="facial_expression_recognition":
		# 	labels = self.classifier_3(features)
		else:
			print("Task name not valid.")
			raise NotImplementedError
		if not self.fm_distillation:
			return labels
		else:
			return labels, features
