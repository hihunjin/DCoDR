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


def get_condition_config(dataset_name: str, dataset_attr_names=None):
    if dataset_name == "two_class_color_mnist":
        TRAIN_CONDITION_CONFIG = TEST_CONDITION_CONFIG = {
            "01234": True,
            "red": True,
        }
    elif dataset_name == "multi_color_mnist":
        TRAIN_CONDITION_CONFIG = TEST_CONDITION_CONFIG = {
            "label": True,
            "color": True,
        }
    elif dataset_name == "waterbirds":
        TRAIN_CONDITION_CONFIG = TEST_CONDITION_CONFIG = dict(zip(["y", "place"], [False, False]))
    elif dataset_name == "celeba":
        assert dataset_attr_names is not None
        _temp = {
            dataset_attr_names[15]: True,  # Eyeglasses
            dataset_attr_names[39]: True,  # Young

            # dataset_attr_names[22]: True,  # Mustache
            # dataset_attr_names[31]: True,  # Smiling
        }
        TRAIN_CONDITION_CONFIG = TEST_CONDITION_CONFIG = _temp


    return TRAIN_CONDITION_CONFIG, TEST_CONDITION_CONFIG


def generate_imgs_classes_contents(train: bool = True):
    from datasets.builder import build_dataset

    dataset_name = "waterbirds"
    dataset = build_dataset(
        _target_=dataset_name,
        split="train" if train else "test",
    )
    # if train:
    #     from datasets import build_dataset, make_condition, make_subset

    #     train_condition_config, _ = get_condition_config(dataset_name)
    #     train_condition = make_condition(dataset.attr_names, train_condition_config)

    #     dataset = make_subset(dataset, train_condition)
    imgs = []
    classes = []
    contents = []
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0].resize((64, 64))
        imgs.append(np.array(img))
        classes.append(np.array(data[1][0]))
        contents.append(np.array(data[1][1]))
    return np.array(imgs), np.array(classes), np.array(contents)


if __name__ == '__main__':
    train_imgs, train_classes, train_contents = generate_imgs_classes_contents()
    train_classes, train_contents = train_contents, train_classes

    train_n_classes = len(np.unique(train_classes))
    np.savez(join(save_folder, f'waterbirds2_train.npz'),
             imgs=train_imgs,
             contents=train_contents,
             classes=train_classes,
             n_classes=train_n_classes,
             )

    test_imgs, test_classes, test_contents = generate_imgs_classes_contents(train=False)

    test_n_classes = len(np.unique(test_classes))
    np.savez(join(save_folder, f'waterbirds2_test.npz'),
             imgs=test_imgs,
             contents=test_contents,
             classes=test_classes,
             n_classes=test_n_classes,
             )
