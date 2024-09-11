import torch.nn as nn


class MLP(nn.Module):
	def __init__(self, num_labels):
		super(MLP, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(14*1024, 1024),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=1024),
			nn.Linear(1024, 128),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=128),
			nn.Linear(128, num_labels)
		)


	def forward(self, x):
		x = self.layers(x)

		return x
