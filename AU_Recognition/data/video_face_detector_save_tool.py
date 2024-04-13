import os
import numpy as np
import pandas as pd
import json
import sys
import argparse
import cv2
import time
import mediapipe as mp

## Saurabh Chatterjee | chatterjeesaurabh38@gmail.com  ##

video_file = 'Saurabh_1.mp4'          #########

dataset_folder = 'AU_Recognition/data'
image_list = []
video_path = os.path.join(dataset_folder, video_file)


def save_as_json(label_dict_list, output_dir):
    with open(output_dir, 'w') as f:
        json.dump(label_dict_list, output_dir)


def rect_to_bb(rect):       # rect: box rectangle produced by a dlib detector
	# take a bounding predicted by dlib and convert it to the format (x, y, w, h) as we would normally do with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)


def shape_to_numpy(shape):        # dlib face landmark detector will return a shape object containing the 68 (x, y)-coordinates of the facial landmark regions
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype='int')
	# loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (int(shape.part(i).x), int(shape.part(i).y))
	# return the list of (x, y)-coordinates
	return coords

def shape_to_list(shape):        # dlib face landmark detector will return a shape object containing the 68 (x, y)-coordinates of the facial landmark regions
	# initialize the list of (x, y)-coordinates
	coords = []
	# loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords.append([int(shape.part(i).x), int(shape.part(i).y)])
	# return the list of (x, y)-coordinates
	return coords



cap = cv2.VideoCapture(video_path)

prev_frame_time = 0
frame_num = 0

# Using MediaPipe Face Detector to detect face:
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

with mp_face_detection.FaceDetection(
     model_selection=1, min_detection_confidence=0.5) as face_detection:

    while True:
        ret, frame = cap.read()
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=0)   # alpha - Scaling Factor (Contrast),   beta - Offset (Brightness)(-25 to make pupil absolute intensity)

        if not ret:
            break
        

        frame_num += 1      ##
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # cv2 takes BGR images - Convert to RGB for mediapipe
        img_h, img_w = frame.shape[:2]

        results = face_detection.process(rgb_frame)   # Activate MediaPipe face_detection  # Output is Normalised bw 0-1

        for (i, detection) in enumerate(results.detections):
            box = detection.location_data.relative_bounding_box

            x, y = int(box.xmin * img_w), int(box.ymin * img_h)         # Since Output was Normalised bw 0-1
            w, h = int(box.width * img_w), int(box.height * img_h)

            # CROP the FACE IMAGE from the Frame:
            face_cropped_img = frame[y:y+int(h*1.05), x:x+int(w*1.05)]      # **Increased Rectangle size by 5% to include face chin portion

            video_file_name = video_file.split('.')[0]      
            video_folder = os.path.join(dataset_folder, video_file_name)        # Store all the video frames as images in this folder
            frame_image_store_path = os.path.join(video_folder, video_file_name + '_' + str(frame_num) + '.jpg')    ## IMAGE LOCATION**

            if not os.path.exists(video_folder):        # if folder not present, create the folder
                os.makedirs(video_folder)

            cv2.imwrite(frame_image_store_path, face_cropped_img)      # Store the video Frame as Image in the folder
        
            image_list.append(frame_image_store_path)   
            

         
        # For FPS Display:
        new_frame_time = time.time()
        time_diff = new_frame_time-prev_frame_time
        fps = 1/(time_diff)
        prev_frame_time = new_frame_time

        # converting the fps into integer
        fps = int(fps)
        # converting the fps to string so that we can display it on frame by using putText function
        fps = str(fps)

        # putting the FPS count on the frame
        cv2.putText(face_cropped_img, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('img', face_cropped_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
## -------------------------------------------------------------------------

# Store all video frame Images Location in a CSV File

# csv_file = os.path.join(dataset_folder, video_file_name + '_csv' + '.csv')
csv_file = os.path.join(dataset_folder, 'data.csv')

df = pd.DataFrame()
df['image_path'] = image_list

df.to_csv(csv_file, index=False)

print('Images location data successfully saved in the CSV File: data.csv')








