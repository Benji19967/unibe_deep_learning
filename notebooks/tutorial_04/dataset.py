from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets.mnist import MNIST


class Digits(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        download: bool = True,
        transforms: Optional[Callable] = None,
    ):
        super(Digits, self).__init__()

        # Construct your MNIST instance here
        self.mnist = MNIST(
            root=root, train=train, download=download, transform=transforms
        )

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, int]:
        """
        Returns the datapoint at index = idx.
        You need to implement this method in such a way
        that the ith element of the Digits class
        is a pair of subsequent MNIST dataset samples.
        That is if MNIST is [a, b, c, d], then Digits
        are [[a, b], [c, d]] and the label is mod 10
        sum of the MNIST labels.

        :param idx: Index of the datapoint.
        :return: 2 image tensors, their mod 10 sum.
        """

        ##### Write your code here #####
        image_0, label_0 = self.mnist[2 * idx]
        image_1, label_1 = self.mnist[2 * idx + 1]

        return (image_0, image_1, (label_0 + label_1) % 10)
        ################################

    def __len__(self) -> int:
        """
        Returns the length of the dataset. Equals half the length of the MNIST dataset.

        :return: The total number of datapoints in the dataset.
        """

        ##### Write your code here #####
        return len(self.mnist) // 2
        ################################
