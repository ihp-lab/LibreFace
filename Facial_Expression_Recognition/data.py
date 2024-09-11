import os
import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

class MyDataset(data.Dataset):
	def __init__(self, data, data_root, csv_file, img_size, train):
		self.data = data
		self.data_root = data_root
		self.csv_file = csv_file
		self.img_size = img_size
		self.train = train

		self.file_list = pd.read_csv(csv_file)
		self.images = self.file_list['image_path']
		self.labels = self.file_list['expression']

		self.convert = transforms.ToTensor()
		self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	def __getitem__(self, index):
		image_path = self.images[index]
		image_path = image_path.split('/')[1]
		image_name = os.path.join(self.data_root, self.data, 'image',image_path)
		image = Image.open(image_name).resize((self.img_size[0], self.img_size[1]))
		image = self.convert(image)
		image = self.transform(image)

		label = self.labels[index]

		return image, label

	def collate_fn(self, data):
		images, labels = zip(*data)

		images = torch.stack(images)
		labels = torch.LongTensor(labels)

		return images, labels

	def __len__(self):
		return len(self.images)
