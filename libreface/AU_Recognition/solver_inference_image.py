import os
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms

from libreface.AU_Recognition.models.resnet18 import ResNet18
from libreface.AU_Recognition.models.mae import MaskedAutoEncoder
from libreface.utils import download_weights


class image_test(object):
    def __init__(self, img_size=256, crop_size=224):
        self.img_size = img_size
        self.crop_size = crop_size

    def __call__(self, img):
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)

        return img
    
class AU_Recognition_Dataset(data.Dataset):
    def __init__(self, frames_path_list, config):
        self.config = config
        self.img_size = config.image_size
        self.crop_size = config.crop_size
        
        self.transform = image_test(img_size=self.img_size, crop_size=self.crop_size)

        self.images = frames_path_list
        

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def __getitem__(self, index):
        image_path = self.images[index]
        image = self.pil_loader(image_path)

        image = self.transform(image)

        return image

    def collate_fn(self, data):
        images = torch.stack(data)

        return images

    def __len__(self):
        return len(self.images)

class solver_inference_image(nn.Module):
    def __init__(self, config):
        super(solver_inference_image, self).__init__()
        self.config = config

        # Setup number of labels

        self.config.num_labels = 12
        self.num_labels = self.config.num_labels

        self.image_transform = image_test(img_size=config.image_size, crop_size=config.crop_size)

        self.device = config.device
        # Initiate the networks
        if config.model_name == "resnet":
            self.model = ResNet18(config).to(self.device)
        elif config.model_name == "emotionnet_mae":
            self.model = MaskedAutoEncoder(config).to(self.device)
        else:
            raise NotImplementedError

   
        if self.config.half_precision:
            print("Use Half Precision.")
   
        # Setup AU index
        self.aus = [1,2,4,5,6,9,12,15,17,20,25,26]


    def pil_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def transform_image_inference(self, aligned_image_path):

        image = self.pil_loader(aligned_image_path)
        # print(image.shape)
        image = self.image_transform(image)

        return image


    def image_inference(self, transformed_image):
        with torch.no_grad():
            self.eval()
            input_image = torch.unsqueeze(transformed_image, 0).to(self.device)
            if self.config.half_precision:
                input_image = input_image.half()
                self.model = self.model.half()
            labels_pred = self.model(input_image)
            if self.config.half_precision:
                labels_pred = labels_pred.float()
            labels_pred = torch.clamp(labels_pred * 5.0, min=0.0, max=5.0)
            return labels_pred
        
    def video_inference(self, dataloader):
        all_labels_pred = None
        with torch.no_grad():
            self.eval()
            for input_image in dataloader:
                input_image = input_image.to(self.device)
                if self.config.half_precision:
                    input_image = input_image.half()
                    self.model = self.model.half()
                labels_pred = self.model(input_image)
                if self.config.half_precision:
                    labels_pred = labels_pred.float()
                labels_pred = torch.clamp(labels_pred * 5.0, min=0.0, max=5.0)
                if all_labels_pred is None:
                    all_labels_pred = labels_pred
                else:
                    all_labels_pred = torch.cat((all_labels_pred, labels_pred), dim = 0)
        return all_labels_pred


    def load_best_ckpt(self):
        download_weights(self.config.weights_download_id, self.config.ckpt_path)
        checkpoints = torch.load(self.config.ckpt_path, map_location=self.device, weights_only=True)['model']
        self.model.load_state_dict(checkpoints, strict=True)

    def run(self, aligned_image_path):
        if "cuda" in self.device:
            torch.backends.cudnn.benchmark = True

        # Test model
        self.load_best_ckpt()
        transformed_image = self.transform_image_inference(aligned_image_path)
        pred_labels = self.image_inference(transformed_image)
        pred_labels = pred_labels.squeeze().tolist()
        return dict(zip(self.aus, pred_labels))
    
    def run_video(self, aligned_image_path_list):

        if "cuda" in self.device:
            torch.backends.cudnn.benchmark = True

        # Test model
        self.load_best_ckpt()
        
        dataset = AU_Recognition_Dataset(aligned_image_path_list, self.config)
        loader = DataLoader(
                    dataset=dataset,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                    shuffle=False,
                    collate_fn=dataset.collate_fn,
                    drop_last=False)

        pred_labels = self.video_inference(loader)
        pred_labels = pred_labels.tolist()
        return self.aus, pred_labels
