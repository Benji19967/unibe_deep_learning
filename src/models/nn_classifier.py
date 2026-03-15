import numpy as np


class NNClassifierVectorized:
    def __init__(self):
        self.images = []
        self.labels = []

    def train(self, images, labels):
        # This is a lazy classifier, so we just save the data.

        self.images = images
        self.labels = labels

    def predict(self, image):
        # 4D Tensor of images: NxHxWxC
        #
        # N - number of images
        # H - height
        # W - width
        # C - number of channels (RGB)
        #
        # self.images   [N, H, W, C]
        # image         [N, H, W, C]

        distances = np.mean(np.abs(self.images - image), axis=(1, 2, 3))  # [N]
        closest_index = np.argmin(distances)
        predicted_label = self.labels[closest_index]
        closest_image = self.images[closest_index]
        min_distance = np.min(distances)

        return predicted_label, min_distance, closest_image
