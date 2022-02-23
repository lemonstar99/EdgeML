import argparse
import torch
import pandas as pd
import numpy as np
import os.path

from scipy.io import arff
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset
from glob import glob
from datetime import datetime



class BaseADDataset(ABC):
    """Anomaly detection dataset base class."""

    def __init__(self, root: str):
        super().__init__()
        self.root = root  # root path to data

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = None  # tuple with original class labels that define the normal class
        self.outlier_classes = None  # tuple with original class labels that define the outlier class

        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.test_set = None  # must be of type torch.utils.data.Dataset

    @abstractmethod
    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__

class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers)
        return train_loader, test_loader

class HR_Dataset(TorchvisionDataset):

    def __init__(self, root:str, normal_class):

        super().__init__(root)
        self.normal_class = normal_class

        # x_array = [[[0 for k in range(3)] for j in range(11932)]]

        # load lists of participant ids
        # id_fb, id_nfb = load_id('/workspace/HR_WearablesData/')
        # id_fb = np.load("/workspace/fitbit_id.npy")
        # id_nfb = np.load("/workspace/nonfitbit_id.npy")
        # id_anomalies = load_labels('/workspace/datasets/Health New Labeling.xlsx')

        # df = load_fitbit_data(id_fb[0])
        # x_array = cut_to_same_length(df, x_array)
        # y_array = np.zeros(x_array.shape[0])
        # index_array = np.arange(x_array.shape[0])

        print("start")
        dim1_train = pd.read_csv("/workspace/dim1_train_short.txt").to_numpy()
        dim2_train = pd.read_csv("/workspace/dim2_train_short.txt").to_numpy()
        dim3_train = pd.read_csv("/workspace/dim3_train_short.txt").to_numpy()

        dim1_test = pd.read_csv("/workspace/dim1_test_short.txt").to_numpy()
        dim2_test = pd.read_csv("/workspace/dim2_test_short.txt").to_numpy()
        dim3_test = pd.read_csv("/workspace/dim3_test_short.txt").to_numpy()

        labels_train = pd.read_csv("/workspace/labels_train_short.csv").to_numpy()
        labels_test = pd.read_csv("/workspace/labels_test_short.csv").to_numpy()
        print("all files loaded.")

        print("train set: ")
        print(dim1_train.shape)
        print(dim2_train.shape)
        print(dim3_train.shape)
        print(len(labels_train))

        print("test set: ")
        print(dim1_test.shape)
        print(dim2_test.shape)
        print(dim3_test.shape)
        print(len(labels_test))

        index_array_train = np.arange(len(labels_train))
        index_array_test = np.arange(len(labels_test))

        x_array_train = np.dstack([dim1_train, dim2_train, dim3_train])
        x_array_test = np.dstack([dim1_test, dim2_test, dim3_test])
        print("creating datasets...")

        train_set = TensorDataset(torch.Tensor(x_array_train), torch.Tensor(labels_train), torch.Tensor(index_array_train))
        self.train_set = train_set

        test_set = TensorDataset(torch.Tensor(x_array_test), torch.Tensor(labels_test), torch.Tensor(index_array_test))
        self.test_set = test_set
        print("done.")