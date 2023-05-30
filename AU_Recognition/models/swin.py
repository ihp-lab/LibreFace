import torch
import torch.nn as nn

from models.swin_transformer import swin_transformer_base,swin_transformer_tiny


class SwinTransformer(nn.Module):
	def __init__(self, opts):
		super(SwinTransformer, self).__init__()

		self.encoder = swin_transformer_tiny(pretrained=False,device=opts.device)
		self.classifier = nn.Sequential(
			# nn.Linear(49*1024, 512),
			nn.Linear(37632,512),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=512),
			nn.Dropout(p=opts.dropout),
			nn.Linear(512, opts.num_labels),
			# nn.ReLU(),
			)

	def forward(self, images):
		batch_size = images.shape[0]
		features = self.encoder(images).reshape(batch_size, -1)
		labels = self.classifier(features)

		return labels


class SwinTransformerMCD(nn.Module):
	def __init__(self, opts):
		super(SwinTransformerMCD, self).__init__()

		self.encoder = swin_transformer_base()
		self.classifier1 = nn.Sequential(
			nn.Linear(49*1024, 512),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=512),
			nn.Dropout(p=opts.dropout),
			nn.Linear(512, opts.num_labels))
		self.classifier2 = nn.Sequential(
			nn.Linear(49*1024, 512),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=512),
			nn.Dropout(p=opts.dropout),
			nn.Linear(512, opts.num_labels))

	def forward_prime(self, images):
		batch_size = images.shape[0]
		features = self.encoder(images).reshape(batch_size, -1)
		labels_1 = self.classifier1(features)
		labels_2 = self.classifier2(features)

		return labels_1, labels_2

	def forward(self, images):
		batch_size = images.shape[0]
		features = self.encoder(images).reshape(batch_size, -1)
		labels = self.classifier1(features)

		return labels


class SwinTransformerM3SDA(nn.Module):
	def __init__(self, opts):
		super(SwinTransformerM3SDA, self).__init__()

		self.encoder = swin_transformer_base()
		self.classifier1 = nn.Sequential(
			nn.Linear(49*1024, 512),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=512),
			nn.Dropout(p=opts.dropout),
			nn.Linear(512, opts.num_labels))
		self.classifier2 = nn.Sequential(
			nn.Linear(49*1024, 512),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=512),
			nn.Dropout(p=opts.dropout),
			nn.Linear(512, opts.num_labels))
		self.classifier3 = nn.Sequential(
			nn.Linear(49*1024, 512),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=512),
			nn.Dropout(p=opts.dropout),
			nn.Linear(512, opts.num_labels))
		self.classifier4 = nn.Sequential(
			nn.Linear(49*1024, 512),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=512),
			nn.Dropout(p=opts.dropout),
			nn.Linear(512, opts.num_labels))

	def forward_combine(self, images):
		batch_size = images.shape[0]
		features = self.encoder(images).reshape(batch_size, -1)
		labels_1 = self.classifier1(features)
		labels_2 = self.classifier2(features)

		return features, labels_1, labels_2

	def forward_1(self, images):
		batch_size = images.shape[0]
		features = self.encoder(images).reshape(batch_size, -1)
		labels_1 = self.classifier1(features)
		labels_2 = self.classifier2(features)

		return features, labels_1, labels_2

	def forward_2(self, images):
		batch_size = images.shape[0]
		features = self.encoder(images).reshape(batch_size, -1)
		labels_3 = self.classifier3(features)
		labels_4 = self.classifier4(features)

		return features, labels_3, labels_4

	def forward(self, images):
		batch_size = images.shape[0]
		features = self.encoder(images).reshape(batch_size, -1)
		labels = self.classifier1(features)

		return labels
