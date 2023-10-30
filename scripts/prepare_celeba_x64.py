from os.path import join as join

import cv2
from tqdm import tqdm
from abc import ABC, abstractmethod
import os
import imageio

import numpy as np
import pandas as pd
import face_alignment  # library from https://github.com/1adrianb/face-alignment

from sklearn.model_selection import train_test_split

dataset_folder = '/NFS/database_personal/anomaly_detection/data/CelebA/celeba'
save_folder = 'cache/preprocess'
os.makedirs(save_folder, exist_ok=True)

np.random.seed(2)


class DataSet(ABC):

    def __init__(self, base_dir=None):
        super().__init__()
        self._base_dir = base_dir

    @abstractmethod
    def read_images(self):
        pass


import csv
import torch
from typing import Optional, List, Union


def _load_csv(
    base_folder,
    filename: str,
    header: Optional[int] = None,
) -> pd.DataFrame:
    with open(os.path.join(base_folder, filename)) as csv_file:
        data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

    if header is not None:
        headers = data[header]
        data = data[header + 1 :]
    else:
        headers = []

    indices = [row[0] for row in data]
    data = [row[1:] for row in data]
    data_int = [list(map(int, i)) for i in data]

    df = pd.DataFrame(data_int, index=indices, columns=headers[1:])
    return df


class CelebA(DataSet):

    def __init__(self, base_dir, train):
        super().__init__(base_dir)

        suffix = 'train'
        if not train:
            suffix = 'test'
        self.__imgs_dir = base_dir
        self.__identity_map_path = os.path.join(self._base_dir, 'identity_CelebA.txt')
        # self.__full_contents = os.path.join(self._base_dir, 'full_contents.csv')

    def __list_imgs(self):
        with open(self.__identity_map_path, 'r') as fd:
            lines = fd.read().splitlines()

        img_paths = []
        img_ids = []
        class_ids = []

        for line in lines:
            img_name, class_id = line.split(' ')
            if not os.path.exists(os.path.join(self.__imgs_dir, img_name)):
                continue
            img_path = os.path.join(self.__imgs_dir, os.path.splitext(img_name)[0] + '.jpg')

            img_paths.append(img_path)
            class_ids.append(class_id)
            img_ids.append(os.path.splitext(img_name)[0] + '.jpg')

        return img_paths, class_ids, img_ids

    def read_images(self, crop_size=(128, 128), target_size=(64, 64)):
        img_paths, class_ids, img_ids = self.__list_imgs()
        full_contents = _load_csv(dataset_folder, "list_attr_celeba.txt", header=1)
        # full_contents = pd.read_csv(self.__full_contents)
        # full_contents = full_contents.set_index('image_id')
        cur_contents = full_contents.loc[img_ids, :].values

        unique_class_ids = list(set(class_ids))

        imgs = np.empty(shape=(len(img_paths), 64, 64, 3), dtype=np.uint8)
        classes = np.empty(shape=(len(img_paths),), dtype=np.uint32)
        contents = cur_contents

        for i in tqdm(range(len(img_paths))):
            img = imageio.imread(img_paths[i])

            if crop_size:
                img = img[
                      (img.shape[0] // 2 - crop_size[0] // 2):(img.shape[0] // 2 + crop_size[0] // 2),
                      (img.shape[1] // 2 - crop_size[1] // 2):(img.shape[1] // 2 + crop_size[1] // 2)
                      ]

            if target_size:
                img = cv2.resize(img, dsize=target_size)

            imgs[i] = img
            classes[i] = unique_class_ids.index(class_ids[i])

        return imgs, classes, contents


def find_landmarks(imgs):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    landmarks = np.zeros(shape=(imgs.shape[0], 68 * 2), dtype=np.float32)
    for i in tqdm(range(imgs.shape[0])):
        preds = fa.get_landmarks(imgs[i])
        if preds is None or len(preds) == 0:
            continue
        landmarks[i] = preds[0].flatten()

    return landmarks

if __name__ == '__main__':
    train_data = CelebA(base_dir=dataset_folder, train='train')
    train_imgs, train_classes, train_contents = train_data.read_images()
    train_landmarks = find_landmarks(train_imgs)

    train_n_classes = len(np.unique(train_classes))
    np.savez(join(save_folder, f'celeba_x64_train.npz'),
             imgs=train_imgs,
             contents=train_contents,
             classes=train_classes,
             n_classes=train_n_classes,
             landmarks=train_landmarks
             )

    test_data = CelebA(base_dir=dataset_folder, train='test')
    test_imgs, test_classes, test_contents = test_data.read_images()
    test_landmarks = find_landmarks(test_imgs)

    test_n_classes = len(np.unique(test_classes))
    np.savez(join(save_folder, f'celeba_x64_test.npz'),
             imgs=test_imgs,
             contents=test_contents,
             classes=test_classes,
             n_classes=test_n_classes,
             landmarks=test_landmarks
             )
