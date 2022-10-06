import os
from pathlib import Path

from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms

from external_library import InceptionResnetV1, MTCNN


class FolderDataset():
    def __init__(self, cfg):
        self.cfg_for_DatasetFromNumpy = cfg["DatasetFromNumpy"]

        pretrained = cfg["pretrained"]
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"device is {self.device}")
        self.face_feature_extractor = InceptionResnetV1(
            pretrained=pretrained,
            device=self.device
        ).eval()

        image_size = cfg["image_size"]
        print(f'image size: {image_size}')
        self.face_detector = MTCNN(
            image_size=image_size,
            margin=0,
            keep_all=False,
            min_face_size=40,
            device=self.device
        )

        self.source_path = './'+cfg['folder_name_for_source']
        self.path_for_image = os.path.join(
            self.source_path,
            cfg['folder_name_for_images']
        )
        print(f'Loading faces from {self.path_for_image}')
        self.image_dataset = datasets.ImageFolder(self.path_for_image)
        self.idx_to_class = {i: c for c, i
                             in self.image_dataset.class_to_idx.items()}

        for i, name in self.idx_to_class.items():
            print(i, name)

        data_length = len(self.image_dataset)
        print(f"data length: {data_length}")

    def setFilePath(
            self,
            source_path, subfolder_name,
            file_name, extension_name
    ):
        source_path = os.path.join(source_path, subfolder_name)
        if not os.path.exists(source_path):
            os.makedirs(source_path)

        file_name_emb = file_name + '_emb' + extension_name
        file_name_lb = file_name + '_lb' + extension_name
        self.path_emb = os.path.join(source_path, file_name_emb)
        self.path_lb = os.path.join(source_path, file_name_lb)

        print(f'path_emb: {self.path_emb}')
        print(f'path_lb: {self.path_lb}')

    def createNumpyData(self):
        def collate_fn(x):
            return x[0]

        dataloader = DataLoader(self.image_dataset,
                                collate_fn=collate_fn)

        np_emb = []
        np_lb = []
        for i, (img, lb) in enumerate(dataloader):
            print(f'[{i:4}]converting to numpy...')

            face, prob = self.face_detector(img, return_prob=True)

            emb = self.face_feature_extractor(
                face.unsqueeze(0).to(self.device)
            )
            emb = np.squeeze(emb.to('cpu').detach().numpy())
            np_emb.append(emb)
            np_lb.append(lb)

        np_emb = np.asarray(np_emb)
        np_lb = np.asarray(np_lb)

        return np_emb, np_lb

    def saveToNumpy(self, np_emb, np_lb):
        print(np_emb.shape)
        print(np_lb.shape)

        try:
            with open(self.path_emb, 'wb') as f:
                np.save(f, np_emb)
            with open(self.path_lb, 'wb') as f:
                np.save(f, np_lb)
        except FileExistsError:
            print(f'saving numpy failed !')

    def loadFromNumpy(self):
        with open(self.path_emb, 'rb') as f:
            np_emb = np.load(f)
        with open(self.path_lb, 'rb') as f:
            np_lb = np.load(f)

        print(np_emb.shape)
        print(np_lb.shape)

        return np_emb, np_lb

    def get_path_to_load_numpy(self):
        return self.path_emb, self.path_lb

    def select_cls_np(
            self,
            np_emb,
            np_lb,
            selected_cls: list
    ):
        print('[Before slicing data]')
        print(f'emb shape: {np_emb.shape}', end=' ')
        print(f'lb shape: {np_lb.shape}')

        condition = np.isin(np_lb, selected_cls)
        np_emb = np_emb[condition]
        np_lb = np_lb[condition]

        print('[After slicing data')
        print(f'emb shape: {np_emb.shape}', end=' ')
        print(f'lb shape: {np_lb.shape}')

        return np_emb, np_lb

    def createOpensetDataset(self, np_emb, np_lb, kkc: list, uuc: list):
        kkc_np_emb, kkc_np_lb = self.select_cls_np(
            np_emb, np_lb, kkc
        )
        uuc_np_emb, uuc_np_lb = self.select_cls_np(
            np_emb, np_lb, uuc
        )


class DatasetFromNumpy(Dataset):
    def __init__(self, np_emb, np_lb):
        self.np_emb = np_emb
        self.np_lb = np_lb

        print(self.np_emb.shape)
        print(self.np_lb.shape)

    def __len__(self):
        return len(self.np_lb)

    def __getitem__(self, idx):

        emb = self.np_emb[idx]
        label = self.np_lb[idx]

        return (emb, label)


def buildDataLoaders(
    dataset,
    batch_size,
    ratio_train_data=0.8,
    ratio_val_data=0.2
):

    dataset_size = len(dataset)
    train_size = int(dataset_size * ratio_train_data)
    val_size = int(dataset_size * ratio_val_data)
    test_size = dataset_size - train_size - val_size

    print(
        "dataset length: ({}) = tr ({}) + val ({}) + tt ({})".format(
            dataset_size, train_size, val_size, test_size
        )
    )

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # output type has to be dictionary
    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }
    # dataset_sizes = {"train": train_size,
    #                  "val": val_size,
    #                  "test": test_size}

    return dataloaders


def saveDataloaders(source_folder, dataloaders):
    source_path = os.path.join('./', source_folder)

    if not os.path.exists(source_path):
        os.makedirs(source_path)

    for phase, dataloader in dataloaders.items():
        file_name = 'dataloader_'+phase+'.pt'
        path = os.path.join(source_path, file_name)
        torch.save(dataloader, path)
        print(f'saved in {path}')
