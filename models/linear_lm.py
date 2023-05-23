import torch.nn as nn
import torch
import torchvision
import pdb

class MLP(nn.Module):
	def __init__(self, opts):
		super(MLP, self).__init__()
		self.dropout = opts.dropout
		self.classifier = nn.Sequential(
				nn.Linear(144, 128),
				nn.ReLU(),
				nn.BatchNorm1d(num_features=128),
				nn.Dropout(p=self.dropout),
				nn.Linear(128, opts.num_labels),
    			nn.Sigmoid()
				)
				

   
	def forward(self, landmarks):
		batch_size = landmarks.shape[0]
		labels = self.classifier(landmarks.reshape(batch_size,-1))
		return labels

