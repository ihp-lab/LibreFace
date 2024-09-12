from libreface.detect_mediapipe_image import *
from libreface.AU_Detection.inference import detect_action_units
from libreface.AU_Recognition.inference import get_au_intensities
from libreface.Facial_Expression_Recognition.inference import get_facial_expression

def get_facial_attributes_image(image_path, temp_dir="./tmp", device="cpu"):
    aligned_image_path = get_aligned_image(image_path, temp_dir=temp_dir)
    detected_aus = detect_action_units(aligned_image_path, device = device)
    au_intensities = get_au_intensities(aligned_image_path, device = device)
    facial_expression = get_facial_expression(aligned_image_path, device = device)
    return {"detected_aus": detected_aus,
            "au_intensities": au_intensities,
            "facial_expression": facial_expression}

