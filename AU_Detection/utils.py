import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from data import MyDataset, MyDataset_GH_Feat, MyDataset_with_lm

def set_seed(seed):
	# Reproducibility
	torch.manual_seed(seed)
	# torch.cuda.manual_seed_all(seed)
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = True

	random.seed(seed)
	np.random.seed(seed)


def get_data_loader(csv_file, train, config):
	dataset = MyDataset(csv_file, train, config)
	loader = DataLoader(
				dataset=dataset,
				batch_size=config.batch_size,
				num_workers=config.num_workers,
				shuffle=train,
				collate_fn=dataset.collate_fn,
				drop_last=train)

	return loader

def get_data_loader_landmark(csv_file, train, config):
	dataset = MyDataset_with_lm(csv_file, train, config)
	loader = DataLoader(
				dataset=dataset,
				batch_size=config.batch_size,
				num_workers=config.num_workers,
				shuffle=train,
				collate_fn=dataset.collate_fn,
				drop_last=train)

	return loader

def get_data_loader_gh(csv_file, train, config):
	dataset = MyDataset_GH_Feat(csv_file, config)
	loader = DataLoader(
				dataset=dataset,
				batch_size=config.batch_size,
				num_workers=config.num_workers,
				shuffle=train,
				collate_fn=dataset.collate_fn,
				drop_last=train)

	return loader

class ForeverDataIterator:
	def __init__(self, data_loader: DataLoader):
		self.data_loader = data_loader
		self.iter = iter(self.data_loader)

	def __next__(self):
		try:
			data = next(self.iter)
		except StopIteration:
			self.iter = iter(self.data_loader)
			data = next(self.iter)
		return data

	def __len__(self):
		return len(self.data_loader)


def mean_status(feats):
	total_mean = np.mean(feats)
	intraclass_mean = list(map(lambda x: np.mean(x), feats))
	interclass_mean = [np.mean(list(map(lambda x: x[i], feats))) for i in range(len(feats[0]))]
	return total_mean, intraclass_mean, interclass_mean


def degree_of_freedoms(feats):
	interclass_df = len(feats) - 1
	intraclass_df = len(feats[0]) - 1
	error_df = intraclass_df * interclass_df
	return interclass_df, intraclass_df, error_df


def cal_total_SS(feats, total_mean):
	total_ss = 0
	for i in range(len(feats)):
		for j in range(len(feats[0])):
			total_ss = total_ss + np.square(feats[i][j] - total_mean)
	return total_ss


def cal_inter_SS(total_mean, intraclass_mean, intra_length):
	inter_ss = 0
	for m in intraclass_mean:
		inter_ss = inter_ss + intra_length * np.square(m - total_mean)
	return inter_ss


def cal_intra_SS(total_mean, interclass_mean, inter_length):
	intra_ss = 0
	for m in interclass_mean:
		intra_ss = intra_ss + inter_length * np.square(m - total_mean)
	return intra_ss


def cal_MS(feats, total_mean, intraclass_mean, interclass_mean):
	total_ss = cal_total_SS(feats=feats,
							total_mean=total_mean)
	inter_ss = cal_inter_SS(total_mean=total_mean,
							intraclass_mean=intraclass_mean,
							intra_length=len(feats[0]))
	intra_ss = cal_intra_SS(total_mean=total_mean,
							interclass_mean=interclass_mean,
							inter_length=len(feats))
	error_ss = total_ss - inter_ss - intra_ss
	interclass_df, intraclass_df, error_df = degree_of_freedoms(feats=feats)
	inter_ms = inter_ss / interclass_df
	intra_ms = intra_ss / intraclass_df
	error_ms = error_ss / error_df
	return inter_ms, intra_ms, error_ms


def cal_status(feats):
	total_mean, intraclass_mean, interclass_mean = mean_status(feats)
	inter_ms, intra_ms, error_ms = cal_MS(feats=feats,
										  total_mean=total_mean,
										  intraclass_mean=intraclass_mean,
										  interclass_mean=interclass_mean)
	return inter_ms, intra_ms, error_ms


def ICC(feats):
	_, intra_ms, error_ms = cal_status(feats=feats)
	numerator = intra_ms -  error_ms
	denominator = intra_ms + (len(feats) - 1) * error_ms
	icc3 = numerator / denominator

	return icc3
