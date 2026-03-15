import os
from pathlib import Path

import numpy as np
from PIL import Image

DOG_DIR = Path("data/raw/PetImages/Dog")
CAT_DIR = Path("data/raw/PetImages/Cat")

N_IMGS = 1200
TARGET_SIZE = (128, 128)  # Easier to compare images of the same size


def load_images():
    dog_paths = os.listdir(DOG_DIR)
    cat_paths = os.listdir(CAT_DIR)

    dog_paths.sort()
    cat_paths.sort()

    dog_paths = [
        os.path.join(DOG_DIR, dog_path)
        for dog_path in dog_paths
        if dog_path.endswith(".jpg")
    ]
    cat_paths = [
        os.path.join(CAT_DIR, cat_path)
        for cat_path in cat_paths
        if cat_path.endswith(".jpg")
    ]

    def load_and_format(image_paths, target_size):
        return [
            np.array(Image.open(path).convert("RGB").resize(target_size))
            for path in image_paths
        ]

    # Load 1200 images for each class
    dog_images = load_and_format(dog_paths[:N_IMGS], TARGET_SIZE)
    cat_images = load_and_format(cat_paths[:N_IMGS], TARGET_SIZE)

    return dog_images, cat_images


def get_train_and_test_datasets(dog_images, cat_images):
    dog_images = np.array(dog_images).astype(np.float32)
    cat_images = np.array(cat_images).astype(np.float32)

    n_train = 1100  # 1100 out of 1200 for training, rest for test
    dog_train = dog_images[:n_train]
    cat_train = cat_images[:n_train]

    dog_test = dog_images[n_train:]
    cat_test = cat_images[n_train:]

    x_train = np.concatenate((dog_train, cat_train), axis=0)
    x_test = np.concatenate((dog_test, cat_test), axis=0)

    y_train = ["dog" for _ in range(len(dog_train))]
    y_train.extend(["cat" for _ in range(len(cat_train))])

    y_test = ["dog" for _ in range(len(dog_test))]
    y_test.extend(["cat" for _ in range(len(cat_test))])

    return x_train, x_test, y_train, y_test
