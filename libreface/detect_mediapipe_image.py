import cv2
import numpy as np
import os
import numpy as np
import scipy.ndimage
from PIL import Image, ImageDraw
from tqdm import tqdm
import mediapipe as mp
import argparse

from libreface.utils import uniquify_file, restructure_landmark_mediapipe

def image_align(img, face_landmarks, output_size=256,
        transform_size=512, enable_padding=True, x_scale=1,
        y_scale=1, em_scale=0.1, alpha=False, pad_mode='const'):

  lm = np.array(face_landmarks)
  lm[:,0] *= img.size[0]
  lm[:,1] *= img.size[1]

  lm_eye_right      = lm[0:16]  
  lm_eye_left     = lm[16:32]  
  lm_mouth_outer   = lm[32:]  
  # lm_mouth_inner   = lm[60 : 68]  # left-clockwise
  lm_mouth_outer_x = lm_mouth_outer[:,0].tolist()
  left_index = lm_mouth_outer_x.index(min(lm_mouth_outer_x))
  right_index = lm_mouth_outer_x.index(max(lm_mouth_outer_x))
  # print(left_index,right_index)
  # Calculate auxiliary vectors.
  eye_left     = np.mean(lm_eye_left, axis=0)
  # eye_left[[0,1]] = eye_left[[1,0]]
  eye_right    = np.mean(lm_eye_right, axis=0)
  # eye_right[[0,1]] = eye_right[[1,0]]
  eye_avg      = (eye_left + eye_right) * 0.5
  eye_to_eye   = eye_right - eye_left
  # print(lm_mouth_outer)s
  mouth_avg    = (lm_mouth_outer[left_index,:] + lm_mouth_outer[right_index,:])/2.0
  # mouth_avg[[0,1]] = mouth_avg[[1,0]]
  
  eye_to_mouth = mouth_avg - eye_avg
  # Choose oriented crop rectangle.
  x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
  x /= np.hypot(*x)
  x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
  x *= x_scale
  y = np.flipud(x) * [-y_scale, y_scale]
  c = eye_avg + eye_to_mouth * em_scale
  quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
  qsize = np.hypot(*x) * 2

  # Shrink.
  shrink = int(np.floor(qsize / output_size * 0.5))
  if shrink > 1:
    rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
    img = img.resize(rsize, Image.ANTIALIAS)
    quad /= shrink
    qsize /= shrink

  # Crop.
  border = max(int(np.rint(qsize * 0.1)), 3)
  crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
  crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
  if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
    img = img.crop(crop)
    quad -= crop[0:2]

  # Pad.
  pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
  pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
  if enable_padding and max(pad) > border - 4:
    pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
    if pad_mode == 'const':
      img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'constant', constant_values=0)
    else:
      img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
    h, w, _ = img.shape
    y, x, _ = np.ogrid[:h, :w, :1]
    mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
    blur = qsize * 0.02
    img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
    img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
    img = np.uint8(np.clip(np.rint(img), 0, 255))
    if alpha:
      mask = 1-np.clip(3.0 * mask, 0.0, 1.0)
      mask = np.uint8(np.clip(np.rint(mask*255), 0, 255))
      img = np.concatenate((img, mask), axis=2)
      img = Image.fromarray(img, 'RGBA')
    else:
      img = Image.fromarray(img, 'RGB')
    quad += pad[:2]

  img = img.transform((transform_size, transform_size), Image.Transform.QUAD,
            (quad + 0.5).flatten(), Image.Resampling.BILINEAR)

  out_image = img.resize((output_size, output_size), Image.Resampling.LANCZOS)
  out_image = img

  return out_image


def uniquify_dir(dir_path):
  dir_path = dir_path.rstrip("/")
  dir_name = dir_path.split("/")[-1]
  par_dir = "/".join(dir_path.split("/")[:-1])

  counter=1
  while os.path.exists(dir_path):
    dir_path = f"{par_dir}/{dir_name}_{counter}"
    counter+=1
  return dir_path

def uniquify_file(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def get_aligned_image(image_path, temp_dir = "./tmp", verbose=False):

  # temp_dir = uniquify_dir(temp_dir)
  os.makedirs(temp_dir, exist_ok=True)

  image_name = ".".join(image_path.split("/")[-1].split(".")[:-1])

  img_path = image_path
  land_save_path = uniquify_file(os.path.join(temp_dir, f"{image_name}_landmark.npy"))
  aligned_img_save_path = uniquify_file(os.path.join(temp_dir, f"{image_name}_aligned.png"))
  annotated_image_save_path = uniquify_file(os.path.join(temp_dir, f"{image_name}_annotated.png"))

  image = cv2.imread(img_path)
  # mp_face_detection = mp.solutions.face_detection
  mp_face_mesh = mp.solutions.face_mesh
  # help(mp_face_detection.FaceDetection)

  # mp_drawing = mp.solutions.drawing_utils 
  # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
  # # mp_drawing = mp.solutions.drawing_utils 
  # mp_drawing_styles = mp.solutions.drawing_styles

  FACEMESH_LIPS = [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                            (17, 314), (314, 405), (405, 321), (321, 375),
                            (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                            (37, 0), (0, 267),
                            (267, 269), (269, 270), (270, 409), (409, 291),
                            (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                            (14, 317), (317, 402), (402, 318), (318, 324),
                            (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                            (82, 13), (13, 312), (312, 311), (311, 310),
                            (310, 415), (415, 308)]
  FACEMESH_LEFT_EYE = [(263, 249), (249, 390), (390, 373), (373, 374),
                                (374, 380), (380, 381), (381, 382), (382, 362),
                                (263, 466), (466, 388), (388, 387), (387, 386),
                                (386, 385), (385, 384), (384, 398), (398, 362)]


  FACEMESH_RIGHT_EYE = [(33, 7), (7, 163), (163, 144), (144, 145),
                                  (145, 153), (153, 154), (154, 155), (155, 133),
                                  (33, 246), (246, 161), (161, 160), (160, 159),
                                  (159, 158), (158, 157), (157, 173), (173, 133)]
  Left_eye = []
  Right_eye = []
  Lips = []
  for (x,y) in FACEMESH_LEFT_EYE:
    if x not in Left_eye:
      Left_eye.append(x)
    if y not in Left_eye:
      Left_eye.append(y)
  # print(Left_eye)
  # print(FACEMESH_LEFT_EYE)

  for (x,y) in FACEMESH_RIGHT_EYE:
    if x not in Right_eye:
      Right_eye.append(x)
    if y not in Right_eye:
      Right_eye.append(y)
  # print(Right_eye)
  # print(FACEMESH_RIGHT_EYE)

  for (x,y) in FACEMESH_LIPS:
    if x not in Lips:
      Lips.append(x)
    if y not in Lips:
      Lips.append(y)
  # print(Lips)
  # print(FACEMESH_LIPS)

  landmark_dict = {}

  image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  with mp_face_mesh.FaceMesh(
      static_image_mode=True,
      refine_landmarks=True,
      max_num_faces=2,
      min_detection_confidence=0.5) as face_mesh:
  #   for name, image in images.items():
      # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
      results = face_mesh.process(image_rgb)
      if results == None:
        print("Processing face mesh had some issue...")
      # Draw face landmarks of each face.
      # print(f'Face landmarks of {name}:')
      if not results.multi_face_landmarks:
        print("Processing landmarks did not result on anything...")
      img_h, img_w, img_c = image.shape
      face_3d = []
      face_2d = []
      
      for face_landmarks in results.multi_face_landmarks:
        
        landmark_dict = restructure_landmark_mediapipe(face_landmarks.landmark)

        for idx, lm in enumerate(face_landmarks.landmark):
          if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:

              x, y = int(lm.x * img_w), int(lm.y * img_h)

              # Get the 2D Coordinates
              face_2d.append([x, y])

              # Get the 3D Coordinates
              face_3d.append([x, y, lm.z])       
          
        # Convert it to the NumPy array
        face_2d = np.array(face_2d, dtype=np.float64)

        # Convert it to the NumPy array
        face_3d = np.array(face_3d, dtype=np.float64)

        # The camera matrix
        focal_length = 1 * img_w

        cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                [0, focal_length, img_w / 2],
                                [0, 0, 1]])

        # The distortion parameters
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Solve PnP
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        # Get rotational matrix
        rmat, jac = cv2.Rodrigues(rot_vec)

        # Get angles
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        # Get the y rotation degree
        x = angles[0] * 360
        y = angles[1] * 360
        z = angles[2] * 360

        pitch = x
        yaw = y
        roll = z

        head_pose = f"pitch:{pitch:.1f}, yaw:{yaw:.1f}, roll:{roll:.1f}"

        # See where the user's head tilting
        # if y < -10:
        #     text = "Looking Left"
        # elif y > 10:
        #     text = "Looking Right"
        # elif x < -10:
        #     text = "Looking Down"
        # elif x > 10:
        #     text = "Looking Up"
        # else:
        #     text = "Forward"

        # Display the nose direction
        # nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

        # p1 = (int(nose_2d[0]), int(nose_2d[1]))
        # p2 = (int(nose_2d[0] + y * 10) , int(nose_2d[1] - x * 10))
        
        # cv2.line(annotated_image, p1, p2, (255, 0, 0), 3)

        # Add the text on the image
        # cv2.putText(annotated_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        # cv2.putText(annotated_image, head_pose, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # cv2.putText(annotated_image, "x: " + str(np.round(x,2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(annotated_image, "y: " + str(np.round(y,2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # cv2.putText(annotated_image, "z: " + str(np.round(z,2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # mp_drawing.draw_landmarks(
        #   image=annotated_image,
        #   landmark_list=face_landmarks,
        #   connections=mp_face_mesh.FACEMESH_TESSELATION,
        #   landmark_drawing_spec=None,
        #   connection_drawing_spec=mp_drawing_styles
        #   .get_default_face_mesh_tesselation_style())
        # mp_drawing.draw_landmarks(
        #   image=annotated_image,
        #   landmark_list=face_landmarks,
        #   connections=mp_face_mesh.FACEMESH_CONTOURS,
        #   landmark_drawing_spec=None,
        #   connection_drawing_spec=mp_drawing_styles
        #   .get_default_face_mesh_contours_style())
        # mp_drawing.draw_landmarks(
        #   image=annotated_image,
        #   landmark_list=face_landmarks,
        #   connections=mp_face_mesh.FACEMESH_IRISES,
        #   landmark_drawing_spec=None,
        #   connection_drawing_spec=mp_drawing_styles
        #   .get_default_face_mesh_iris_connections_style())

        lm_left_eye_x = []
        lm_left_eye_y = []
        lm_right_eye_x = []
        lm_right_eye_y = []
        lm_lips_x = []
        lm_lips_y = []
        for i in Left_eye:
          lm_left_eye_x.append(face_landmarks.landmark[i].x)
          lm_left_eye_y.append(face_landmarks.landmark[i].y)
        for i in Right_eye:
          lm_right_eye_x.append(face_landmarks.landmark[i].x)
          lm_right_eye_y.append(face_landmarks.landmark[i].y)
        for i in Lips:
          lm_lips_x.append(face_landmarks.landmark[i].x)
          lm_lips_y.append(face_landmarks.landmark[i].y)
        lm_x = lm_left_eye_x + lm_right_eye_x + lm_lips_x
        lm_y = lm_left_eye_y + lm_right_eye_y + lm_lips_y
        landmark = np.array([lm_x,lm_y]).T

  # if verbose:
  #   print("Aligned Image save to: ",aligned_img_save_path)
  #   print("Annotated Image save to: ",annotated_image_save_path)
  # cv2.imwrite(annotated_image_save_path,annotated_image)
  # pdb.set_trace()

  aligned_image = image_align(Image.fromarray(image_rgb), landmark)
  aligned_image.save(aligned_img_save_path)

  head_pose = {"pitch":pitch,
               "yaw":yaw,
               "roll":roll}
  
  return aligned_img_save_path, head_pose, landmark_dict

def get_aligned_video_frames(frames_df, temp_dir="./tmp"):
  aligned_frames_paths = []
  head_pose_list = []
  landmark_list = []
  for _, row in tqdm(frames_df.iterrows(), desc="Aligning face for video frames..."):
      aligned_image_path, head_pose, landmark_dict = get_aligned_image(row["path_to_frame"], temp_dir)
      aligned_frames_paths.append(aligned_image_path)
      head_pose_list.append(head_pose)
      landmark_list.append(landmark_dict)
  
  return aligned_frames_paths, head_pose_list, landmark_list

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_path', type=str, default='/home/achaubey/Desktop/projects/data/DISFA/images/LeftVideoSN001_comp/LeftVideoSN001_comp_0001.png', help='Input path to input images')
  
  args = parser.parse_args()
  aligned_image_path, _, _ = get_aligned_image(args.image_path)
  print("Aligned image saved to ", aligned_image_path)
