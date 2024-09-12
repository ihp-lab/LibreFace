import gdown
import os

# Define a function to download the model weights
def download_weights(drive_id, model_path):
    model_dir = "/".join(model_path.split("/")[:-1])
    os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(model_path):
        print(f"Downloading model weights - {model_path}...")
        gdown.download(id=drive_id, output=model_path)
        if not os.path.exists(model_path):
            print("Error occured in downloading...")
    else:
        print(f"{model_path} already exists. Skippind model weights download.")

    return model_path
