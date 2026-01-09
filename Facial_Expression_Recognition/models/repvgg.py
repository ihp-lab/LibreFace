import torch.nn as nn
import torch
import torchvision
import pdb
import timm

class RepVGG(nn.Module):
    def __init__(self, opts):
        super(RepVGG, self).__init__()
        self.fm_distillation = opts.fm_distillation
        self.dropout = opts.dropout
        # self.add_landmark = opts.add_landmark
        # self.add_proj_layer = opts.proj_layer
        out_dim = 512
        
        repvgg = timm.create_model("repvgg_a0", pretrained=True)
        vgg_classifier = repvgg.get_classifier()             
        last_out = vgg_classifier.out_features              
        clf_name = _find_module_name(repvgg, vgg_classifier)
        appended = nn.Sequential(vgg_classifier, nn.Linear(last_out, out_dim))
        _set_module_by_name(repvgg, clf_name, appended)

        self.encoder = repvgg  

        # if not self.add_landmark:
        self.classifier = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Dropout(p=self.dropout),
                nn.Linear(128, opts.num_labels),
                nn.Sigmoid()
                )
                

        # else:
        # 	if self.add_proj_layer:
        # 		self.proj_layer = nn.Linear(68*2, 512)
        # 		self.classifier = nn.Sequential(
        # 			nn.Linear(1024, 128),
        # 			nn.ReLU(),
        # 			nn.BatchNorm1d(num_features=128),
        # 			nn.Dropout(p=self.dropout),
        # 			nn.Linear(128, opts.num_labels)),
        # 	else:
        # 		self.classifier = nn.Sequential(
        # 			nn.Linear(68*2+512, 128),
        # 			nn.ReLU(),
        # 			nn.BatchNorm1d(num_features=128),
        # 			nn.Dropout(p=self.dropout),
        # 			nn.Linear(128, opts.num_labels)),
   
    def forward(self, images, landmarks=None):
        batch_size = images.shape[0]
        features = self.encoder(images).reshape(batch_size, -1)
        # if landmarks is not None:
        # 	landmarks = landmarks.reshape(batch_size, -1)
        # 	if not self.add_proj_layer:
        # 		landmarks -= landmarks.min()
        # 		landmarks /= landmarks.max()
   
        # if not self.add_landmark:
        labels = self.classifier(features)
        # else:
        # 	if not self.add_proj_layer:
        # 		labels = self.classifier(torch.cat([features,landmarks],dim=1))
        # 	else:
        # 		labels = self.classifier(torch.cat([features,self.proj_layer(landmarks)],dim=1))
        # pdb.set_trace()
        if not self.fm_distillation:
            return labels
        else:
            return labels, features


# Chatgpt
def _find_module_name(model: nn.Module, target: nn.Module) -> str:
    """Return dotted path name of `target` inside `model`."""
    for name, module in model.named_modules():
        if module is target:
            return name
    raise RuntimeError("Classifier module not found via named_modules().")

def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Set nested attribute `name` (e.g., 'head.fc') on `model` to `new_module`."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)