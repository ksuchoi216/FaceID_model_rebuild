import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class FaceRecognizer(nn.Module):
    def __init__(self, cfg):
        super(FaceRecognizer, self).__init__()

        input_size = cfg["input_size"]
        output_size = cfg["output_size"]
        self.classify = cfg["classify"]

        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.fc1(x)
        # print(x.shape)
        # print(x)
        x = F.softmax(x, dim=1)
        # print(x)
        return x


def build_model(cfg):
    model = FaceRecognizer(cfg)
    return model


def load_dataloader(phase):
    source_path = './data'
    file_name = 'dataloader_'+phase+'.pt'
    path_for_dataloader = os.path.join(source_path, file_name)
    dataloader = torch.load(path_for_dataloader)
    return dataloader


'''
class Builder:
    def __init__(self, cfg):
        self.pretrained = cfg["pretrained"]
        self.num_classes = cfg["num_classes"]
        self.classify = cfg["classify"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"device is {self.device}")

        self.face_feature_extractor = InceptionResnetV1(
            pretrained=self.pretrained,
            num_classes=self.num_classes,
            classify=self.classify,
            device=self.device,
        )

        # Transfer Learning
        for param in self.face_feature_extractor.parameters():
            param.requires_grad = False
            # print(f'{param.requires_grad}')

        for param in self.face_feature_extractor.logits.parameters():
            param.requires_grad = True
            # print(f'{param.requires_grad}')

        print("Loading model was just completed.")

    def setModel(self, pretrained, num_classes, classify):
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.classify = classify

        self.face_feature_extractor = InceptionResnetV1(
            pretrained=self.pretrained,
            num_classes=self.num_classes,
            classify=self.classify,
            device=self.device,
        )

    def getModel(self, device=None):
        if device is None:
            device = self.device

        return self.face_feature_extractor.to(device)

    def loadModel(self, path_for_saved_model, device):
        self.face_feature_extractor.load_state_dict(
            torch.load(path_for_saved_model, map_location=torch.device(device))
        )
        self.face_feature_extractor.to(device)
        self.face_feature_extractor.eval()

        return self.face_feature_extractor

    def summary(self):
        print(self.face_feature_extractor)

class Builder_Seperated_Model:
    def __init__(self, cfg):
        self.pretrained = cfg["pretrained"]
        self.num_classes = cfg["num_classes"]
        self.classify = cfg["classify"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"device is {self.device}")

        face_feature_extractor = InceptionResnetV1(
            pretrained=self.pretrained,
            num_classes=self.num_classes,
            classify=self.classify,
            device=self.device,
        )

        self.classifier = Classifier(
            cfg["classifier"], face_feature_extractor=face_feature_extractor
        ).to(self.device)

        print("Loading model was just completed.")

    def getModel(self):
        return self.classifier

    def summary(self):
        # summary(self.face_feature_extractor, size)
        print(self.classifier)
'''