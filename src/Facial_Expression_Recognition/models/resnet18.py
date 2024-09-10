import torch.nn as nn
import torch
import torchvision
import pdb

class ResNet(nn.Module):
	def __init__(self, opts):
		super(ResNet, self).__init__()
		self.fm_distillation = opts.fm_distillation
		self.dropout = opts.dropout
		resnet18 = torchvision.models.resnet18(pretrained=True)
		resnet18_layers = list(resnet18.children())[:-1]
		self.encoder = nn.Sequential(*resnet18_layers)
		self.classifier = nn.Sequential(
				nn.Linear(512, 128),
				nn.ReLU(),
				nn.BatchNorm1d(num_features=128),
				nn.Dropout(p=self.dropout),
				nn.Linear(128, opts.num_labels),
				nn.Sigmoid()
    )
   
	def forward(self, images):
		batch_size = images.shape[0]
		features = self.encoder(images).reshape(batch_size, -1)
		labels = self.classifier(features)
		if not self.fm_distillation:
			return labels
		else:
			return labels, features
