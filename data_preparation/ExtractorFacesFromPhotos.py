# import cv2
import os
import warnings

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets

from external_library import MTCNN


warnings.filterwarnings("ignore")


class ExtractorFacesFromPhotos(object):
    def __init__(self, cfg):
        folder_for_raw = cfg['folder_for_raw']
        self.path_for_photos = os.path.join(
            folder_for_raw,
            cfg["folder_for_photos"]
        )
        self.path_for_faces = os.path.join(
            folder_for_raw,
            cfg["folder_to_save_faces"]
        )
        self.image_size = cfg["image_size"]

        self.mtcnn = MTCNN(
            image_size=self.image_size,
            margin=0,
            keep_all=False,
            min_face_size=20,
        )  # keep_all=False
        self.mtcnn_show = MTCNN(select_largest=False, post_process=False)

        if not os.path.exists(self.path_for_faces):
            os.makedirs(self.path_for_faces)
            print("new folder was created in ", self.path_for_faces)

        self.FaceThreshold = cfg['FaceThreshold']
        print(f'FaceThreshold: {self.FaceThreshold}')

    def save_cropped_faces(self, isShowImage=False, ImageInterval=30):
        print("Starting data load...")

        image_dataset = datasets.ImageFolder(self.path_for_photos)
        idx_to_class = {
            i: c for c, i in image_dataset.class_to_idx.items()
        }  # accessing names of peoples from folder names
        for i, name in idx_to_class.items():
            print(i, name)
            path_ = os.path.join(
                self.path_for_faces,
                name
            )
            if not os.path.exists(path_):
                os.makedirs(path_)

        def collate_fn(x):
            return x[0]

        data_loader = DataLoader(image_dataset, collate_fn=collate_fn)

        print("data loader is created: ", data_loader)
        img_num = 0
        current_idx = 0
        for i, (img, idx) in enumerate(data_loader):
            # print(img, idx)
            if current_idx != idx:
                current_idx += 1
                img_num = 0

            if i % ImageInterval == 0 and isShowImage is True:
                print("=" * 50)
                print("class: ", idx)
                face = self.mtcnn_show(img)
                face = face.permute(1, 2, 0).int().numpy()
                if face is not None:
                    plt.imshow(face)
                    plt.axis("off")
                    plt.show()
                print("=" * 50)

            name = idx_to_class[idx]
            file_name = str(img_num) + ".png"
            save_path = os.path.join(
                self.path_for_faces,
                name,
                file_name
            )

            face, prob = self.mtcnn(img, return_prob=True)
            if face is not None and prob >= self.FaceThreshold:
                _ = self.mtcnn_show(img, save_path=save_path)
                print(
                    f"[{img_num:3}][{prob:4f}] saved face image in {save_path}")

                img_num += 1
