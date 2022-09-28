# import torch
# import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
# from torchvision.utils import save_image


def splitDataLoaders(dataset,
                     batch_size: int,
                     ratios: dict) -> (dict, dict):
    train_ratio = ratios['train']
    val_ratio = ratios['val']
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
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
    dataset_sizes = {"train": train_size,
                     "val": val_size,
                     "test": test_size}

    return dataloaders, dataset_sizes


def splitDatasetBasedOnSelection(dataset, unselected_labels: list):
    selected_list = []
    unselected_list = []

    for img, label in dataset:
        if label in unselected_labels:
            unselected_list.append([img.numpy(), label])
        else:
            selected_list.append([img.numpy(), label])

    unselected_numpy = np.asarray(unselected_list)
    selected_numpy = np.asarray(selected_list)

    return selected_numpy, unselected_numpy


def build_dataloader(cfg: dict, isSplit: bool, dataset):
    batch_size = cfg['batch_size']
    ratio = {'train': cfg['train_ratio'], 'val': cfg['val_ratio']}

    if isSplit:
        return splitDataLoaders(dataset, batch_size, ratio)
    else:
        return DataLoader(dataset, batch_size, shuffle=True)


class NumpyBaseDataset(Dataset):
    def __init__(
        self,
        cfg: dict,
        numpy_dataset: np.ndarray
    ):
        image_size = cfg["image_size"]
        image_rotation_angle = cfg["image_rotation_angle"]
        isTransformer = cfg["isTransformer"]

        self.dataset = numpy_dataset
        self.data_length = numpy_dataset.shape[0]

        if isTransformer is True:
            self.transformer = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation((0, image_rotation_angle)),
                    transforms.ToTensor(),  # convert a PIL image or ndarray to tensor
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    ),  # normalize to Imagenet mean and std
                ]
            )

    def __len__(self):
        return self.data_length

    def __getitem__(self, idx):

        label = self.dataset[idx, 1]
        image = self.dataset[idx, 0]
        image = Image.fromarray(image.astype('uint8'), 'RGB')

        if self.transformer:
            image = self.transformer(image)

        return (image, label)


class Folder_Dataset():
    def __init__(self, cfg):
        self.data = cfg["data_path"]
        self.ratios = {'train': cfg["train_ratio"], 'val': cfg["val_ratio"]}
        self.batch_size = cfg["batch_size"]
        self.isSplit = True
        image_size = cfg["image_size"]
        image_rotation_angle = cfg["image_rotation_angle"]
        isTransformer = cfg["isTransformer"]

        print(f'image size: {image_size}')

        if isTransformer is True:
            self.transformer = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation((0, image_rotation_angle)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    ),  # normalize to Imagenet mean and std
                ]
            )
        else:
            self.transformer = None

        self.image_dataset = datasets.ImageFolder(self.data,
                                                  transform=self.transformer)

        self.idx_to_class = {i: c for c, i
                             in self.image_dataset.class_to_idx.items()}

        for i, name in self.idx_to_class.items():
            print(i, name)

        print(f"batch_size: {self.batch_size} \n")
        
    def setIsSplit(self, isSplit):
        self.isSplit = isSplit

    def getImageDataset(self):
        return self.image_dataset

    def createDataLoaders(self):
        if self.isSplit:
            dataloaders, dataset_sizes = splitDataLoaders(self.image_dataset,
                                                          self.batch_size,
                                                          self.ratios)
            return dataloaders, dataset_sizes, self.idx_to_class
        else:
            dataset_sizes = {'test': len(self.image_dataset)}
            dataloader = DataLoader(self.image_dataset,
                                    self.batch_size,
                                    shuffle=True)
            dataloaders = {'test': dataloader}
            return dataloaders, dataset_sizes