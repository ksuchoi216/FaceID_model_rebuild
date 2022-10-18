import os
from pathlib import Path

from PIL import Image
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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

        self.raw_path = './'+cfg['folder_for_raw']
        self.path_for_image = os.path.join(
            self.raw_path,
            cfg['folder_for_images']
        )
        print(f'Loading faces from {self.path_for_image}')

        self.image_dataset = datasets.ImageFolder(self.path_for_image)

        self.idx_to_class = {i: c for c, i
                             in self.image_dataset.class_to_idx.items()}

        for i, name in self.idx_to_class.items():
            print(i, name)

        data_length = len(self.image_dataset)
        # print(f"data length: {data_length}")

    def setFilePath(
            self,
            raw_path, save_folder,
            file_name, extension_name
    ):
        raw_path = os.path.join(raw_path, save_folder)
        if not os.path.exists(raw_path):
            os.makedirs(raw_path)

        file_name_emb = file_name + '_emb' + extension_name
        file_name_lb = file_name + '_lb' + extension_name
        self.path_emb = os.path.join(raw_path, file_name_emb)
        self.path_lb = os.path.join(raw_path, file_name_lb)

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

            if face is not None and prob >= self.face_threshold:
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
        print(f'saved emb:{np_emb.shape} lb:{np_lb.shape}')

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

        print(f'loaded emb:{np_emb.shape} lb:{np_lb.shape}')

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

        print(f'dataset initial emb:{np_emb.shape} lb:{np_lb.shape}')

    def __len__(self):
        return len(self.np_lb)

    def __getitem__(self, idx):

        emb = self.np_emb[idx]
        label = self.np_lb[idx]

        return emb, label


def buildDataLoaders(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    ratio_train=0.8,
    ratio_val=0.2
):
    dataset = DatasetFromNumpy(data, labels)

    dataset_size = len(dataset)
    train_size = int(dataset_size * ratio_train)
    val_size = int(dataset_size * ratio_val)
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

    return dataloaders


def saveDataloaders(source_folder, save_folder,  dataloaders):
    source_folder = './'+source_folder
    source_path = os.path.join(source_folder, save_folder)

    if not os.path.exists(source_path):
        os.makedirs(source_path)

    for phase, dataloader in dataloaders.items():
        file_name = 'dataloader_'+phase+'.pt'
        path = os.path.join(source_path, file_name)
        torch.save(dataloader, path)
        print(f'saved in {path}')


def loadNumpy(source_folder, save_folder, file_name):
    source_path = './'+source_folder
    source_path = os.path.join(source_path, save_folder)
    path = os.path.join(source_path, file_name)

    print(f'loading from {path}')

    try:
        with open(path, 'rb') as f:
            numpy_array = np.load(f)

        return numpy_array
    except FileExistsError:
        print(f'loading numpy failed !')


def saveNumpy(numpy, source_path, file_name):
    path = os.path.join(source_path, file_name)
    print(f'saveing to {path}')

    try:
        with open(path, 'wb') as f:
            np.save(f, numpy)

        return numpy_array
    except FileExistsError:
        print(f'saving numpy failed !')


def filterNumpy(data: np.ndarray, labels: np.ndarray, selected_labels: list):
    mask = [True if label in selected_labels else False for label in labels]
    data_ = data[mask, :]
    labels_ = labels[mask]

    print('\n')
    print('[FilterNumpy]')
    print(f'converted from {data.shape} to {data_.shape}')
    print(f'converted from {labels_.shape} to {labels_.shape}')

    return data_, labels_


def splitNumpy(data: np.ndarray, labels: np.ndarray, ratios: dict):
    res = {}
    print('\n')
    print('[splitNumpy]')
    if 'train' in ratios:
        remain_size = round(1-ratios['train'], 2)
        test_size = round(ratios['test']/remain_size, 2)
        print(f'[train exist] Ratios: remain {remain_size} test {test_size}')
        X_train, X_remain, y_train, y_remain = train_test_split(
            data,
            labels,
            test_size=remain_size
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_remain,
            y_remain,
            test_size=test_size
        )

        res['train'] = [X_train, y_train]
        res['val'] = [X_val, y_val]
        res['test'] = [X_test, y_test]
    else:
        print('[no train exist] There is no train ratio in ratios')
        X_val, X_test, y_val, y_test = train_test_split(
            data,
            labels,
            test_size=ratios['test']
        )
        res['val'] = [X_val, y_val]
        res['test'] = [X_test, y_test]

    print(res.keys())
    return res


def putTogetherData(data_a: dict, data_b: dict):
    phases = ['train', 'val', 'test']
    res = {}

    for phase in phases:
        res[phase] = [0]*2
        if phase in data_a and phase in data_b:
            res[phase][0] = np.vstack((data_a[phase][0], data_b[phase][0]))
            res[phase][1] = np.hstack((data_a[phase][1], data_b[phase][1]))
        elif phase in data_a:
            res[phase][0] = data_a[phase][0]
            res[phase][1] = data_a[phase][1]
        else:
            res[phase][0] = data_b[phase][0]
            res[phase][1] = data_b[phase][1]

    print('\n')
    print('[putTogether]')
    print(res.keys())
    print(res["train"][0].shape, res["train"][1].shape)
    print(res["val"][0].shape, res["val"][1].shape)
    print(res["test"][0].shape, res["test"][1].shape)
    return res


def buildMultipleDataLoaders(
    dict_data: dict,
    batch_size: int
):
    print('\n')
    print('[Build dataloaders]')

    dataloaders = {}
    for phase, data in dict_data.items():
        print(f'[{phase}] labels: {set(data[1])}')
        dataset = DatasetFromNumpy(data[0], data[1])
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        dataloaders[phase] = dataloader

    return dataloaders


def convertLabels(
    labels: np.ndarray,
    selected_labels: list,
    marked_label: int,
    unmarked_label: int
):
    s_lbs = selected_labels
    m_lb = marked_label
    unm_lb = unmarked_label

    res = [m_lb if lb in s_lbs else unm_lb for lb in labels]
    res = np.asarray(res)
    num_m_lb = np.where(res == m_lb, 1, 0).sum()
    num_unm_lb = np.where(res == unm_lb, 1, 0).sum()

    print(f'num_m_lb = {num_m_lb}, num_unm_lb = {num_unm_lb} set: {set(res)}')
    return res
