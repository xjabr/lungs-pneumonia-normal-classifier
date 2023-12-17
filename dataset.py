import os

import torch

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from glob import glob

from PIL import Image, ImageFilter


def image_preprocessing(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    image = image.convert('L')
    image = image.resize((256, 256))
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
    return ~np.asarray(image).flatten()

def image_preprocessing_torch(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    return ~np.asarray(image).flatten()

def load_dataset():
    train_images_normal = glob('datasets/train/NORMAL/*.jpeg') + glob('datasets/test/NORMAL/*.jpeg')
    train_images_pneumonia = glob('datasets/train/PNEUMONIA/*.jpeg') + glob('datasets/test/PNEUMONIA/*.jpeg')
    train_label_normal = [0] * len(train_images_normal)
    train_label_pneumonia = [1] * len(train_images_pneumonia)

    x, y = list(train_images_normal + train_images_pneumonia), list(train_label_normal + train_label_pneumonia)

    df = pd.DataFrame({'image': x[0:15], 'label': y[0:15]})
    df = df.sample(frac=1).reset_index(drop=True)

    print(df.head())

    df["image"] = df["image"].apply(image_preprocessing)

    ss = StandardScaler()
    X_standard = ss.fit_transform(np.stack(df["image"].values, axis=0))

    print("Image processing done!")

    return X_standard, df["label"].values
