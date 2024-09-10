import torch.nn as nn
import torchvision


class ResNet(nn.Module):
	def __init__(self, opts):
		super(ResNet, self).__init__()

		dropout = opts.dropout

		resnet18 = torchvision.models.resnet18(pretrained=True)
		self.encoder_1 = nn.Sequential(*list(resnet18.children())[:-6])
		self.encoder_2 = nn.Sequential(*list(resnet18.children())[-6])
		self.encoder_3 = nn.Sequential(*list(resnet18.children())[-5])
		self.encoder_4 = nn.Sequential(*list(resnet18.children())[-4])
		self.encoder_5 = nn.Sequential(*list(resnet18.children())[-3])

		self.interpreter_1 = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(512, 256, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.BatchNorm2d(num_features=256)
		)
		self.interpreter_2 = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(256, 128, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.BatchNorm2d(num_features=128)
		)
		self.interpreter_3 = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
			nn.Conv2d(128, 64, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.BatchNorm2d(num_features=64)
		)
		self.interpreter_4 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.BatchNorm2d(num_features=64)
		)
		self.interpreter_5 = nn.Sequential(
			nn.Conv2d(64, 32, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.Dropout(p=dropout),
			nn.BatchNorm2d(num_features=32),
			nn.Conv2d(32, opts.num_labels, kernel_size=5, padding=2)
		)

	def forward(self, images):
		f_1 = self.encoder_1(images) # [B, 64, 56, 56]
		f_2 = self.encoder_2(f_1) # [B, 64, 56, 56]
		f_3 = self.encoder_3(f_2) # [B, 128, 28, 28]
		f_4 = self.encoder_4(f_3) # [B, 256, 14, 14]
		f_5 = self.encoder_5(f_4) # [B, 512, 7, 7]

		f_6 = self.interpreter_1(f_5) + f_4 # [B, 256, 14, 14]
		f_7 = self.interpreter_2(f_6) + f_3 # [B, 128, 28, 28]
		f_8 = self.interpreter_3(f_7) + f_2 # [B, 64, 56, 56]
		f_9 = self.interpreter_4(f_8) + f_1 # [B, 64, 56, 56]

		labels = self.interpreter_5(f_9)

		return labels
