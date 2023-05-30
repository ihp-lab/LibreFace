import torch

import os
import cv2
import dlib
import numpy as np
from imutils import face_utils


def findlandmark(img_path):
	cascade = '../face_align/shape_predictor_68_face_landmarks.dat'
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(cascade)

	image = cv2.imread(img_path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 1)

	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		return shape


def expression2heatmapV2(img_path, label, size, sigma):
	lmk_path = img_path.replace('images', 'landmarks')[:-4]+'.npy'
	if os.path.exists(lmk_path):
		lmk = np.load(lmk_path)/4
	else:
		# print('find landmark {}'.format(img_path))
		lmk = findlandmark(img_path)/4 # [68, 2]

	lmk_eye_left = lmk[36:42]
	lmk_eye_right = lmk[42:48]
	eye_left = np.mean(lmk_eye_left, axis=0)
	eye_right = np.mean(lmk_eye_right, axis=0)
	lmk_eyebrow_left = lmk[17:22]
	lmk_eyebrow_right = lmk[22:27]
	eyebrow_left = np.mean(lmk_eyebrow_left, axis=0)
	eyebrow_right = np.mean(lmk_eyebrow_right, axis=0)
	IOD = np.linalg.norm(lmk[42] - lmk[39])

	threshold = 0.5
	heatmap = np.zeros((size, size))
	for i in range(17,68):
		gauss_noise = np.fromfunction(lambda y,x : ((x-lmk[i,0])**2 \
											+ (y-lmk[i,1])**2) / -(2.0*sigma*sigma),
											(size, size), dtype=int)
		gauss_noise = np.exp(gauss_noise)
		gauss_noise[gauss_noise < threshold] = 0
		gauss_noise[gauss_noise > threshold] = 1

		heatmap += gauss_noise
	gauss_noise_1 = np.fromfunction(lambda y,x : ((x-eye_left[0])**2 \
											+ (y-eye_left[1]-IOD)**2) / -(2.0*sigma*sigma),
											(size, size), dtype=int)
	gauss_noise_1 = np.exp(gauss_noise_1)
	gauss_noise_1[gauss_noise_1 < threshold] = 0
	gauss_noise_1[gauss_noise_1 > threshold] = 1
	heatmap += gauss_noise_1
	gauss_noise_2 = np.fromfunction(lambda y,x : ((x-eye_right[0])**2 \
											+ (y-eye_right[1]-IOD)**2) / -(2.0*sigma*sigma),
											(size, size), dtype=int)
	gauss_noise_2 = np.exp(gauss_noise_2)
	gauss_noise_2[gauss_noise_2 < threshold] = 0
	gauss_noise_2[gauss_noise_2 > threshold] = 1
	heatmap += gauss_noise_2

	heatmap = np.clip(heatmap, 0., 1.)*(label+1)

	return heatmap