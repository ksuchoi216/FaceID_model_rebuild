import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader

from external_library import MTCNN, InceptionResnetV1


def load_dataloader(phase, source_folder, save_folder):
    source_folder = './' + source_folder
    source_path = os.path.join(source_folder, save_folder)
    file_name = 'dataloader_'+phase+'.pt'
    path_for_dataloader = os.path.join(source_path, file_name)
    print(path_for_dataloader)
    dataloader = torch.load(path_for_dataloader)
    return dataloader


class FullyConnectedNueralNetwork(nn.Module):
    def __init__(self, cfg):
        super(FaceRecognizer, self).__init__()

        input_size = cfg["input_size"]
        output_size = cfg["output_size"]
        self.classify = cfg["classify"]

        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x


def calculateEuclideanDist(A, B):
    return torch.dist(A, B).item()


def calculateSimilarity(A, B):
    return F.cosine_similarity(A, B).item()


class DistanceBasedModel(object):
    def __init__(self, cfg):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        cfg_f = cfg['face_extractor']
        self.single_face_detector = MTCNN(
            image_size=cfg_f['image_size'],
            margin=0,
            keep_all=False,
            min_face_size=20,
        )
        self.face_feature_extractor = InceptionResnetV1(
            pretrained=cfg_f['pretrained'],
            device=device
        ).eval()

        data_path = './' + cfg['folder_for_data']
        path_for_data = os.path.join(
            data_path, cfg['folder_for_photos']
        )

        self.isCosSimilarity = cfg['isCosSimilarity']
        if self.isCosSimilarity:
            self.dist_fn = calculateSimilarity
        else:
            self.dist_fn = calculateEuclideanDist

        print(f'loading registered photos from {path_for_data}')
        dataset = datasets.ImageFolder(path_for_data)
        idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

        def collate_fn(x):
            return x[0]

        dataloader = DataLoader(dataset, collate_fn=collate_fn)
        self.name_list = []
        self.embedding_list = []

        for img, idx in dataloader:
            face, _ = self.single_face_detector(img, return_prob=True)
            if face is not None:
                emb = self.face_feature_extractor(face.unsqueeze(0).to(device))
                self.embedding_list.append(emb.detach())
                self.name_list.append(idx_to_class[idx])

        print(self.name_list)

    def forward(self, embs):
        batch_size, _ = embs.shape
        # print('batch_size:', batch_size)

        pred_list = []
        for i in range(batch_size):
            emb = embs[i, :]
            # print('emb:', emb.shape)

            dist_list = []
            for emb_db in self.embedding_list:
                dist = self.dist_fn(emb, emb_db)
                dist_list.append(round(dist, 3))

            if self.isCosSimilarity:
                target_dist = max(dist_list)
            else:
                target_dist = min(dist_list)

            target_idx = dist_list.index(target_dist)
            # name = self.name_list[target_idx]
            pred_list.append(target_idx)

        print('pred_list:', pred_list)
        return pred_list


def build_model(cfg):
    model_name = cfg['model_name']

    if model_name == "fcnn":
        model = FullyConnectedNeuralNetwork(cfg)
    elif model_name == 'dist':
        model = DistanceBasedModel(cfg)
    else:
        raise ValueError()

    return model
