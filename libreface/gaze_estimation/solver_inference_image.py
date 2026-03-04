import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader

import mediapipe as mp

from libreface.gaze_estimation.models.mlp import MLP
from libreface.utils import download_weights


# MediaPipe landmark indices for eyes + irises used as gaze features
_FACEMESH_LEFT_EYE  = [263, 249, 390, 373, 374, 380, 381, 382, 362,
                        466, 388, 387, 386, 385, 384, 398]
_FACEMESH_RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                        246, 161, 160, 159, 158, 157, 173]
_FACEMESH_IRISES    = [468, 469, 470, 471, 472, 473, 474, 475, 476, 477]

# Feature dimension: (16 + 16 + 10) landmarks * 3 coords (x, y, z) = 126
GAZE_FEAT_DIM = (len(_FACEMESH_LEFT_EYE) + len(_FACEMESH_RIGHT_EYE) + len(_FACEMESH_IRISES)) * 3


def extract_gaze_landmarks(image_path):
	"""Extract eye + iris landmarks from an aligned image using MediaPipe.

	Args:
		image_path (str): Path to an aligned face image.

	Returns:
		np.ndarray of shape (GAZE_FEAT_DIM,) float32, or None if no face detected.
	"""
	img = np.array(Image.open(image_path).convert("RGB"))
	mp_face_mesh = mp.solutions.face_mesh

	with mp_face_mesh.FaceMesh(
		static_image_mode=True,
		refine_landmarks=True,   # required for iris landmarks (468+)
		max_num_faces=1,
		min_detection_confidence=0.5
	) as face_mesh:
		results = face_mesh.process(img)

	if not results.multi_face_landmarks:
		return None

	lm = results.multi_face_landmarks[0].landmark
	indices = _FACEMESH_LEFT_EYE + _FACEMESH_RIGHT_EYE + _FACEMESH_IRISES
	feat = []
	for i in indices:
		feat.extend([lm[i].x, lm[i].y, lm[i].z])

	return np.array(feat, dtype=np.float32)


class Gaze_Detection_Dataset(data.Dataset):
	def __init__(self, frames_path_list, config):
		self.config = config
		self.inputs = frames_path_list

	def __getitem__(self, index):
		feat = extract_gaze_landmarks(self.inputs[index])
		if feat is None:
			feat = np.zeros(self.config.mlp_input_size, dtype=np.float32)
		return torch.from_numpy(feat)

	def collate_fn(self, data):
		return torch.stack(data)

	def __len__(self):
		return len(self.inputs)


class solver_gaze_image(nn.Module):
	def __init__(self, config):
		super(solver_gaze_image, self).__init__()
		self.config = config

		# Setup number of labels (yaw + pitch)
		self.config.num_labels = 2
		self.num_labels = self.config.num_labels

		self.device = config.device

		# Initiate the network
		if config.model_name == "mlp":
			self.model = MLP(config.mlp_input_size, self.num_labels, 256).to(self.device)
		else:
			raise NotImplementedError(f"Unknown model_name: {config.model_name}")

		if self.config.half_precision:
			print("Use Half Precision.")

	def transform_image_inference(self, aligned_image_path):
		feat = extract_gaze_landmarks(aligned_image_path)
		if feat is None:
			feat = np.zeros(self.config.mlp_input_size, dtype=np.float32)
		return torch.from_numpy(feat)

	def image_inference(self, transformed_feat):
		with torch.no_grad():
			self.eval()
			input_feat = torch.unsqueeze(transformed_feat, 0).to(self.device)
			if self.config.half_precision:
				input_feat = input_feat.half()
				self.model = self.model.half()
			pred = self.model(input_feat)
			if self.config.half_precision:
				pred = pred.float()
			return pred

	def video_inference(self, dataloader):
		all_preds = None
		with torch.no_grad():
			self.eval()
			for input_feat in dataloader:
				input_feat = input_feat.to(self.device)
				if self.config.half_precision:
					input_feat = input_feat.half()
					self.model = self.model.half()
				pred = self.model(input_feat)
				if self.config.half_precision:
					pred = pred.float()
				if all_preds is None:
					all_preds = pred
				else:
					all_preds = torch.cat((all_preds, pred), dim=0)
		return all_preds

	def load_best_ckpt(self):
		download_weights(self.config.weights_download_id, self.config.ckpt_path)
		checkpoints = torch.load(self.config.ckpt_path, map_location=self.device, weights_only=True)['model']
		self.model.load_state_dict(checkpoints, strict=True)

	def run(self, aligned_image_path):
		if "cuda" in self.device:
			torch.backends.cudnn.benchmark = True

		self.load_best_ckpt()
		transformed_feat = self.transform_image_inference(aligned_image_path)
		pred = self.image_inference(transformed_feat)
		pred = pred.squeeze().tolist()
		return {"yaw": pred[0], "pitch": pred[1]}

	def run_video(self, aligned_image_path_list):
		if "cuda" in self.device:
			torch.backends.cudnn.benchmark = True

		self.load_best_ckpt()
		dataset = Gaze_Detection_Dataset(aligned_image_path_list, self.config)
		loader = DataLoader(
					dataset=dataset,
					batch_size=self.config.batch_size,
					num_workers=self.config.num_workers,
					shuffle=False,
					collate_fn=dataset.collate_fn,
					drop_last=False)
		preds = self.video_inference(loader)
		preds = preds.tolist()
		return preds