from libreface.detect_mediapipe_image import *
from libreface.AU_Detection.inference import detect_action_units
from libreface.AU_Recognition.inference import get_au_intensities
from libreface.Facial_Expression_Recognition.inference import get_facial_expression

def get_facial_attributes(image_path, temp_dir="./tmp"):
    aligned_image_path = get_aligned_image(image_path, temp_dir=temp_dir)
    