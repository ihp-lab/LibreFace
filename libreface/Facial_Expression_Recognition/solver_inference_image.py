import os
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms

from libreface.Facial_Expression_Recognition.models.resnet18 import ResNet
from libreface.utils import download_weights

class Facial_Expression_Dataset(data.Dataset):
    def __init__(self, frames_path_list, img_size):
        self.img_size = img_size

        self.images = frames_path_list

        self.convert = transforms.ToTensor()
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        image_path = self.images[index]
        image_name = image_path
        image = Image.open(image_name).resize((self.img_size[0], self.img_size[1]))
        image = self.convert(image)
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

        self.device = config.device

        # Initiate the networks
        if config.student_model_name == "resnet":
            self.student_model = ResNet(config).to(self.device)
        else:
            raise NotImplementedError

        self.load_best_ckpt()

        self.img_size = (config.image_size, config.image_size)
        self.convert = transforms.ToTensor()
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def read_and_transform_image(self, image_path):
        
        image = Image.open(image_path).resize((self.img_size[0], self.img_size[1]))
        image = self.convert(image)
        image = self.transform(image)
        return image

    def image_inference(self, transformed_image):
        with torch.no_grad():
            self.student_model.eval()
            input_image = torch.unsqueeze(transformed_image, 0).to(self.device)
            labels_pred, _ = self.student_model(input_image)
            labels_pred = torch.argmax(labels_pred, 1)
            return labels_pred
        
    def video_inference(self, dataloader):
        all_labels_pred = None
        with torch.no_grad():
            self.student_model.eval()
            for input_image in dataloader:
                input_image = input_image.to(self.device)
                labels_pred, _ = self.student_model(input_image)
                labels_pred = torch.argmax(labels_pred, 1)
                if all_labels_pred is None:
                    all_labels_pred = labels_pred
                else:
                    all_labels_pred = torch.cat((all_labels_pred, labels_pred), dim=0)
        return all_labels_pred

    def load_best_ckpt(self):
        download_weights(self.config.weights_download_id, self.config.ckpt_path)
        checkpoints = torch.load(self.config.ckpt_path, map_location=self.device, weights_only=True)['model']
        self.student_model.load_state_dict(checkpoints, strict=True)
  
  
    def run(self, aligned_image_path):
        
        transformed_image = self.read_and_transform_image(aligned_image_path)
        pred_label = self.image_inference(transformed_image)
        pred_label = pred_label.squeeze().tolist()
        return pred_label
    
    def run_video(self, aligned_image_path_list):
        
        dataset = Facial_Expression_Dataset(aligned_image_path_list, (self.config.image_size, self.config.image_size))
        loader = DataLoader(
                    dataset=dataset,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                    shuffle=False,
                    collate_fn=dataset.collate_fn,
                    drop_last=False)
        
        pred_label = self.video_inference(loader)

        pred_label = pred_label.tolist()
        return pred_label

