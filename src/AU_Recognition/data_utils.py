import os
import cv2
import dlib
import torch
import numpy as np
from imutils import face_utils


def findlandmark(img_path):
	cascade = './data/shape_predictor_68_face_landmarks.dat'
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(cascade)

	image = cv2.imread(img_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 1)

	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		return shape


def add_noise(heatmap, channel, label, center_1, center_2, center_3, sigma, size, threshold=0.01):
	if not center_1[0] == -1:
		gauss_noise_1 = np.fromfunction(lambda y,x : ((x-center_1[0])**2 \
												+ (y-center_1[1])**2) / -(2.0*sigma*sigma),
												(size, size), dtype=int)
		gauss_noise_1 = np.exp(gauss_noise_1)
		gauss_noise_1[gauss_noise_1 < threshold] = 0
		gauss_noise_1[gauss_noise_1 > 1] = 1
		gauss_noise_1 = gauss_noise_1 * label[channel].item()
	else:
		gauss_noise_1 = np.zeros((size, size))
	if not center_2[0] == -1:
		gauss_noise_2 = np.fromfunction(lambda y,x : ((x-center_2[0])**2 \
												+ (y-center_2[1])**2) / -(2.0*sigma*sigma),
												(size, size), dtype=int)
		gauss_noise_2 = np.exp(gauss_noise_2)
		gauss_noise_2[gauss_noise_2 < threshold] = 0
		gauss_noise_2[gauss_noise_2 > 1] = 1
		gauss_noise_2 = gauss_noise_2 * label[channel].item()
	else:
		gauss_noise_2 = np.zeros((size, size))
	if not center_3[0] == -1:
		gauss_noise_3 = np.fromfunction(lambda y,x : ((x-center_3[0])**2 \
												+ (y-center_3[1])**2) / -(2.0*sigma*sigma),
												(size, size), dtype=int)
		gauss_noise_3 = np.exp(gauss_noise_3)
		gauss_noise_3[gauss_noise_3 < threshold] = 0
		gauss_noise_3[gauss_noise_3 > 1] = 1
		gauss_noise_3 = gauss_noise_3 * label[channel].item()
	else:
		gauss_noise_3 = np.zeros((size, size))

	heatmap[channel] = np.maximum(np.maximum(gauss_noise_1, gauss_noise_2), gauss_noise_3)

	return heatmap


def au2heatmap(img_path, label, size, config):
	size = size // 4
	lmk_path = img_path.replace('images', 'landmarks')[:-4]+'.npy'
	if os.path.exists(lmk_path):
		lmk = np.load(lmk_path) / 4
	else:
		print('find landmark {}'.format(img_path))
		lmk = findlandmark(img_path) / 4

	heatmap = np.zeros((config.num_labels, size, size))
	sigma = config.sigma

	if config.data == 'BP4D':
		# au6
		heatmap = add_noise(heatmap, 0, label, (lmk[1]+lmk[41]+lmk[31])/3, (lmk[15]+lmk[46]+lmk[35])/3, [-1, -1], sigma, size)

		# au10
		heatmap = add_noise(heatmap, 1, label, lmk[31], lmk[35], lmk[51], sigma, size)

		# au12
		heatmap = add_noise(heatmap, 2, label, lmk[48], lmk[54], [-1, -1], sigma, size)

		# au14
		heatmap = add_noise(heatmap, 3, label, lmk[48], lmk[54], [-1, -1], sigma, size)

		# au17
		heatmap = add_noise(heatmap, 4, label, [-1, -1], [-1, -1], (lmk[57]+lmk[8])/2, sigma, size)
	elif config.data == 'DISFA':
		# au1
		heatmap = add_noise(heatmap, 0, label, lmk[21], lmk[22], (lmk[21]+lmk[22]+lmk[27])/3, sigma, size)

		# au2
		heatmap = add_noise(heatmap, 1, label, lmk[18], lmk[25], [-1, -1], sigma, size)

		# au4
		heatmap = add_noise(heatmap, 2, label, lmk[21], lmk[22], (lmk[21]+lmk[22]+lmk[27])/3, sigma, size)

		# au5
		heatmap = add_noise(heatmap, 3, label, (lmk[37]+lmk[38])/2, (lmk[43]+lmk[44])/2, [-1, -1], sigma, size)

		# au6
		heatmap = add_noise(heatmap, 4, label, (lmk[1]+lmk[41]+lmk[31])/3, (lmk[15]+lmk[46]+lmk[35])/3, [-1, -1], sigma, size)

		# au9
		heatmap = add_noise(heatmap, 5, label, lmk[31], lmk[35], lmk[28], sigma, size)

		# au12
		heatmap = add_noise(heatmap, 6, label, lmk[48], lmk[54], [-1, -1], sigma, size)

		# au15
		heatmap = add_noise(heatmap, 7, label, lmk[48], lmk[54], [-1, -1], sigma, size)

		# au17
		heatmap = add_noise(heatmap, 8, label, [-1, -1], [-1, -1], (lmk[57]+lmk[8])/2, sigma, size)

		# au20
		heatmap = add_noise(heatmap, 9, label, lmk[48], lmk[54], lmk[51], sigma, size)

		# au25
		heatmap = add_noise(heatmap, 10, label, [-1, -1], [-1, -1], (lmk[61]+lmk[64])/2, sigma, size)

		# au26
		heatmap = add_noise(heatmap, 11, label, [-1, -1], [-1, -1], (lmk[61]+lmk[64])/2, sigma, size)

	return heatmap

def heatmap2au(heatmap):
	label = torch.amax(heatmap, dim=(2,3))

	return label
