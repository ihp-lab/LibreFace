import os
import random
import pandas as pd
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import transforms

from data_utils import au2heatmap
import numpy as np

class image_train(object):
	def __init__(self, img_size=256, crop_size=224):
		self.img_size = img_size
		self.crop_size = crop_size

	def __call__(self, img):
		transform = transforms.Compose([
			transforms.Resize(self.img_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])
		])
		img = transform(img)

		return img


class image_test(object):
	def __init__(self, img_size=256, crop_size=224):
		self.img_size = img_size
		self.crop_size = crop_size

	def __call__(self, img):
		transform = transforms.Compose([
			transforms.Resize(self.img_size),
			transforms.CenterCrop(self.crop_size),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])
		])
		img = transform(img)

		return img


class MyDataset(data.Dataset):
	def __init__(self, csv_file, train, config):
		self.config = config
		self.csv_file = csv_file

		self.data = config.data
		self.data_root = config.data_root
		self.img_size = config.image_size
		self.crop_size = config.crop_size
		self.train = train
		if self.train:
			self.transform = image_train(img_size=self.img_size, crop_size=self.crop_size)
		else:
			self.transform = image_test(img_size=self.img_size, crop_size=self.crop_size)

		self.file_list = pd.read_csv(csv_file)
		self.images = self.file_list['image_path']
		if self.data == 'BP4D':
			self.labels = [
							self.file_list['au1'],
							self.file_list['au2'],
							self.file_list['au4'],
							self.file_list['au6'],
							self.file_list['au7'],
							self.file_list['au10'],
							self.file_list['au12'],
							self.file_list['au14'],
							self.file_list['au15'],
							self.file_list['au17'],
							self.file_list['au23'],
							self.file_list['au24'],
						]
		elif self.data == 'DISFA':
			self.labels = [
							self.file_list['au1'],
							self.file_list['au2'],
							self.file_list['au4'],
       						self.file_list['au5'],
							self.file_list['au6'],
							self.file_list['au9'],
							self.file_list['au12'],
							self.file_list['au15'],
							self.file_list['au17'],
							self.file_list['au20'],
							self.file_list['au25'],
							self.file_list['au26']
						]
		self.num_labels = len(self.labels)

	def data_augmentation(self, image, flip, crop_size, offset_x, offset_y):
		image = image[:,offset_x:offset_x+crop_size,offset_y:offset_y+crop_size]
		if flip:
			image = torch.flip(image, [2])

		return image

	def pil_loader(self, path):
		with open(path, 'rb') as f:
			with Image.open(f) as img:
				return img.convert('RGB')

	def __getitem__(self, index):
		image_path = self.images[index]
		image_name = os.path.join(self.data_root, image_path)
		image = self.pil_loader(image_name)

		label = []
		for i in range(self.num_labels):
			label.append(float(self.labels[i][index]))
		label = torch.FloatTensor(label)

		if self.train:
			heatmap = au2heatmap(image_name, label, self.img_size, self.config)
			heatmap = torch.from_numpy(heatmap)
			offset_y = random.randint(0, self.img_size - self.crop_size)
			offset_x = random.randint(0, self.img_size - self.crop_size)
			flip = random.randint(0, 1)
			image = self.transform(image)
			image = self.data_augmentation(image, flip, self.crop_size, offset_x, offset_y)
			heatmap = self.data_augmentation(heatmap, flip, self.crop_size // 4, offset_x // 4, offset_y // 4)

			return image, label, heatmap
		else:
			image = self.transform(image)

			return image, label

	def collate_fn(self, data):
		if self.train:
			images, labels, heatmaps = zip(*data)

			images = torch.stack(images)
			labels = torch.stack(labels).float()
			heatmaps = torch.stack(heatmaps).float()

			return images, labels, heatmaps
		else:
			images, labels = zip(*data)

			images = torch.stack(images)
			labels = torch.stack(labels).float()

			return images, labels

	def __len__(self):
		return len(self.images)



class MyDataset_GH_Feat(data.Dataset):
	def __init__(self, csv_file, config):
		self.config = config
		self.csv_file = csv_file

		self.data = config.data
		self.data_root = config.data_root

		self.file_list = pd.read_csv(csv_file)
		self.images = self.file_list['image_path']

		if self.data == 'BP4D':
			self.labels = [
							self.file_list['au1'],
							self.file_list['au2'],
							self.file_list['au4'],
							self.file_list['au6'],
							self.file_list['au7'],
							self.file_list['au10'],
							self.file_list['au12'],
							self.file_list['au14'],
							self.file_list['au15'],
							self.file_list['au17'],
							self.file_list['au23'],
							self.file_list['au24']
						]
		elif self.data == 'DISFA':
			self.labels = [
							self.file_list['au1'],
							self.file_list['au2'],
							self.file_list['au4'],
       						self.file_list['au5'],
							self.file_list['au6'],
							self.file_list['au9'],
							self.file_list['au12'],
							self.file_list['au15'],
							self.file_list['au17'],
							self.file_list['au20'],
							self.file_list['au25'],
							self.file_list['au26']
						]

		self.num_labels = len(self.labels)


	def __getitem__(self, index):
		image_path = self.images[index]
		feature_path = os.path.join('/home/ICT2000/dchang/TAC_project/data', image_path[:-4]+'.npy')
		feature_path = feature_path.replace('images', 'gh_feat')
		feature = np.load(feature_path)
		feature = torch.from_numpy(feature).view(-1)

		label = []
		for i in range(self.num_labels):
			label.append(int(self.labels[i][index]))
		label = torch.FloatTensor(label)

		return feature, label


	def collate_fn(self, data):
		features, labels = zip(*data)

		features = torch.stack(features)
		labels = torch.stack(labels)

		return features, labels


	def __len__(self):
		return len(self.images)


class MyDataset_with_lm(data.Dataset):
	def __init__(self, csv_file, train, config):
		self.config = config
		self.csv_file = csv_file

		self.data = config.data
		self.data_root = config.data_root
		self.img_size = config.image_size
		self.crop_size = config.crop_size
		self.train = train
		if self.train:
			self.transform = image_train(img_size=self.img_size, crop_size=self.crop_size)
		else:
			self.transform = image_test(img_size=self.img_size, crop_size=self.crop_size)

		self.file_list = pd.read_csv(csv_file)
		self.images = self.file_list['image_path']
		if self.data == 'BP4D':
			self.labels = [
							self.file_list['au6'],
							self.file_list['au10'],
							self.file_list['au12'],
							self.file_list['au14'],
							self.file_list['au17']
						]
		elif self.data == 'DISFA':
			self.labels = [
							self.file_list['au1'],
							self.file_list['au2'],
							self.file_list['au4'],
       						self.file_list['au5'],
							self.file_list['au6'],
							self.file_list['au9'],
							self.file_list['au12'],
							self.file_list['au15'],
							self.file_list['au17'],
							self.file_list['au20'],
							self.file_list['au25'],
							self.file_list['au26']
						]
		self.num_labels = len(self.labels)

	def data_augmentation(self, image, flip, crop_size, offset_x, offset_y):
		image = image[:,offset_x:offset_x+crop_size,offset_y:offset_y+crop_size]
		if flip:
			image = torch.flip(image, [2])

		return image

	def pil_loader(self, path):
		with open(path, 'rb') as f:
			with Image.open(f) as img:
				return img.convert('RGB')

	def __getitem__(self, index):
		image_path = self.images[index]
		image_name = os.path.join(self.data_root, image_path)
		image = self.pil_loader(image_name)

		lm_path = image_path.replace('images', 'landmarks')[:-4]+'.npy'
		lm_name = os.path.join(self.data_root, lm_path)
		landmark = np.load(lm_name)
		landmark = torch.FloatTensor(landmark)
		label = []
		for i in range(self.num_labels):
			label.append(float(self.labels[i][index]))
		label = torch.FloatTensor(label)

		if self.train:
			heatmap = au2heatmap(image_name, label, self.img_size, self.config)
			heatmap = torch.from_numpy(heatmap)
			offset_y = random.randint(0, self.img_size - self.crop_size)
			offset_x = random.randint(0, self.img_size - self.crop_size)
			flip = random.randint(0, 1)
			image = self.transform(image)
			image = self.data_augmentation(image, flip, self.crop_size, offset_x, offset_y)
			heatmap = self.data_augmentation(heatmap, flip, self.crop_size // 4, offset_x // 4, offset_y // 4)
			return image, label, heatmap, landmark
		else:
			image = self.transform(image)

			return image, label, landmark

	def collate_fn(self, data):
		if self.train:
			images, labels, heatmaps, landmarks = zip(*data)

			images = torch.stack(images)
			labels = torch.stack(labels).float()
			heatmaps = torch.stack(heatmaps).float()
			landmarks = torch.stack(landmarks).float()
			return images, labels, heatmaps, landmarks
		else:
			images, labels, landmarks = zip(*data)

			images = torch.stack(images)
			labels = torch.stack(labels).float()
			landmarks = torch.stack(landmarks).float()
			return images, labels, landmarks

	def __len__(self):
		return len(self.images)