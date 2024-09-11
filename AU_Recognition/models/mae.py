import torch
import torch.nn as nn
# import models.modeling_pretrain

from timm.models import create_model
from models.masking_generator import RandomMaskingGenerator


class MaskedAutoEncoder(nn.Module):
	def __init__(self, opts):
		super(MaskedAutoEncoder, self).__init__()
		self.fm_distillation = opts.fm_distillation
		self.hidden_dim = opts.hidden_dim
		self.mae = create_model(
			'pretrain_mae_base_patch16_224',
			pretrained=False,
			drop_path_rate=0.0,
			drop_block_rate=None,
		)

		self.encoder = self.mae.encoder

		self.masked_position_generator = RandomMaskingGenerator((14, 14), 0.)

		self.proj = nn.Linear(196*768, self.hidden_dim)
		
		self.interpreter = nn.Sequential(
			nn.Linear(self.hidden_dim, 32),
			nn.ReLU(),
			nn.BatchNorm1d(num_features=32),
			nn.Dropout(p=opts.dropout),
			nn.Linear(32, opts.num_labels))

	def forward(self, images):
		B, C, H, W = images.shape
		F = 1
		masks = []
		for _ in range(B):
			masks.append(torch.Tensor(self.masked_position_generator()).repeat(F, 1))
		masks = torch.cat(masks, dim=0).to(torch.bool).cuda()

		features = self.encoder(images, masks)
		features = features.reshape(B, 196*768)
		features = self.proj(features).reshape(B, F, self.hidden_dim)
		features = torch.mean(features, dim=1)

		labels = self.interpreter(features)
  
		if not self.fm_distillation:
			return labels
		else:
			return labels, features