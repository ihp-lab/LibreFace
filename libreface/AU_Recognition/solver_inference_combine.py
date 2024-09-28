import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

from libreface.AU_Recognition.models.resnet18_combine import ResNet18
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
    
class Common_Dataset(data.Dataset):
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

class solver_inference_image_task_combine(nn.Module):
    def __init__(self, config):
        super(solver_inference_image_task_combine, self).__init__()
        self.config = config
        self.img_size = config.image_size
        self.crop_size = config.crop_size

        self.device = config.device

        # Setup number of labels
        self.au_recognition_num_labels = self.config.au_recognition_num_labels
        self.au_detection_num_labels = self.config.au_detection_num_labels

        self.image_transform = image_test(img_size=self.img_size, crop_size=self.crop_size)

        # Initiate the networks
        if config.model_name == "resnet":
            self.model = ResNet18(config).to(self.device)

        self.load_best_ckpt()

        # Setup AU index
        self.au_recognition_aus = [1,2,4,5,6,9,12,15,17,20,25,26]
        self.au_detection_aus = [1,2,4,6,7,10,12,14,15,17,23,24]
        
    def pil_loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
            
    def read_and_transform_image(self, image_path):
        image = self.pil_loader(image_path)
        image = self.image_transform(image)
        return image
    
    def image_inference_au_detection(self, transformed_image):
        with torch.no_grad():
            self.eval()
            input_image = torch.unsqueeze(transformed_image, 0).to(self.device)
            pred = self.model(input_image,task_name="au_detection")
            pred = (pred >= 0.5).int()
            return pred
    
    def image_inference_au_recognition(self, transformed_image):
        with torch.no_grad():
            self.eval()
            input_image = torch.unsqueeze(transformed_image, 0).to(self.device)
            labels_pred = self.model(input_image,task_name='au_recognition')
            labels_pred = labels_pred * 5.0
            return labels_pred
        
    def video_inference_au_detection(self, dataloader):
        all_preds = None
        with torch.no_grad():
            self.eval()
            for input_image in dataloader:
                input_image = input_image.to(self.device)
                pred = self.model(input_image, task_name = "au_detection")
                pred = (pred >= 0.5).int()
                if all_preds is None:
                    all_preds = pred
                else:
                    all_preds = torch.cat((all_preds, pred), dim=0)
        return all_preds
    
    def video_inference_au_recognition(self, dataloader):
        all_label_preds = None
        with torch.no_grad():
            self.eval()
            for input_image in dataloader:
                input_image = input_image.to(self.device)
                labels_pred = self.model(input_image,task_name='au_recognition')
                labels_pred = labels_pred * 5.0
                if all_label_preds is None:
                    all_label_preds = labels_pred
                else:
                    all_label_preds = torch.cat((all_label_preds, labels_pred), dim = 0)
        return all_label_preds


    def load_best_ckpt(self):
        download_weights(self.config.weights_download_id, self.config.ckpt_path)
        checkpoints = torch.load(self.config.ckpt_path, map_location=self.device, weights_only=True)['model']
        self.model.load_state_dict(checkpoints, strict=True)

    def run(self, aligned_image_path, task):
        
        transformed_image = self.read_and_transform_image(aligned_image_path)

        if task == "au_recognition":
            pred_labels = self.image_inference_au_recognition(transformed_image)
            pred_labels = pred_labels.squeeze().tolist()
            return dict(zip(self.au_recognition_aus, pred_labels))
        elif task == "au_detection":
            pred_labels =  self.image_inference_au_detection(transformed_image)
            pred_labels = pred_labels.squeeze().tolist()
            return dict(zip(self.au_detection_aus, pred_labels))
        else:
            raise NotImplementedError(f"run() is not defined for the given task = {task}")
        
    def run_video(self, aligned_image_path_list, task):
        
        dataset = Common_Dataset(aligned_image_path_list, self.config)
        loader = DataLoader(
                    dataset=dataset,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                    shuffle=False,
                    collate_fn=dataset.collate_fn,
                    drop_last=False)

        if task == "au_recognition":
            pred_labels = self.video_inference_au_recognition(loader)
            pred_labels = pred_labels.tolist()
            return self.au_recognition_aus, pred_labels
        elif task == "au_detection":
            pred_labels =  self.video_inference_au_detection(loader)
            pred_labels = pred_labels.tolist()
            return self.au_detection_aus, pred_labels
        else:
            raise NotImplementedError(f"run_video() is not defined for the given task = {task}")
            
        
