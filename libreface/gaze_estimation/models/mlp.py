import torch.nn as nn


class MLP(nn.Module):
	def __init__(self, input_size, num_labels, hidden_layer_size):
		super(MLP, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(input_size, hidden_layer_size),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=hidden_layer_size),
			nn.Linear(hidden_layer_size, hidden_layer_size),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=hidden_layer_size),
			nn.Linear(hidden_layer_size, num_labels)
			# add tanh function here
			# nn.Tanh()
		)


	def forward(self, x):
		x = self.layers(x)
		# multiply x by 90 or 180 depending on what the range is
		# x = x * 140
		return x