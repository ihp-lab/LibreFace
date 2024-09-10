import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader

from data import MyDataset

def set_seed(seed):
	# Reproducibility
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

	random.seed(seed)
	np.random.seed(seed)

def get_data_loaders(config):
	num_workers = config.num_workers
	data = config.data
	data_root = config.data_root
	batch_size = config.batch_size
	image_size = config.image_size
	train_csv = config.train_csv
	test_csv = config.test_csv

	train_loader = get_data_loader(data, data_root, os.path.join(data_root, data, train_csv), batch_size, image_size, True, num_workers)
	test_loader = get_data_loader(data, data_root, os.path.join(data_root, data, test_csv), batch_size, image_size, False, num_workers)

	return train_loader, test_loader

def get_data_loader(data, data_root, csv_file, batch_size, image_size, train, num_workers=4):
	dataset = MyDataset(data, data_root, csv_file, (image_size, image_size), train)
	loader = DataLoader(
				dataset=dataset,
				batch_size=batch_size,
				num_workers=num_workers,
				shuffle=train,
				collate_fn=dataset.collate_fn)

	return loader
