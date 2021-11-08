import argparse
import torch
import pandas as pd
import numpy as np

from scipy.io import arff
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, TensorDataset

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

class EP_Dataset(TorchvisionDataset):
    
    def __init__(self, root: str, normal_class):

        super().__init__(root)
        self.n_classes = 2
        self.normal_class = normal_class

        print("normal class: ", normal_class)

        # train set
        #load data file path
        url1_train = 'data/epilepsy/EpilepsyDimension1_TRAIN.arff'
        url2_train = 'data/epilepsy/EpilepsyDimension2_TRAIN.arff'
        url3_train = 'data/epilepsy/EpilepsyDimension3_TRAIN.arff'

        # get x and y as dataframe
        x_dim1_train, target_train = get_data(url1_train)
        x_dim2_train, __ = get_data(url2_train)
        x_dim3_train, __ = get_data(url3_train)

        # combine 3 dimensions of x
        x_train = np.dstack([x_dim1_train, x_dim2_train, x_dim3_train])
        # process output y and produce index
        y_train, index_train = get_target(target_train, normal_class)

        # train only on normal data, extracting normal data
        x_final_train, y_final_train, index_final_train = get_training_set(x_train, y_train, index_train)

        # print("size: ", x_final_train.shape)
        train_set = TensorDataset(torch.Tensor(x_final_train), torch.Tensor(y_final_train), torch.Tensor(index_final_train))
        self.train_set = train_set

        # set up testing set
        url1_test = 'data/epilepsy/EpilepsyDimension1_TEST.arff'
        url2_test = 'data/epilepsy/EpilepsyDimension2_TEST.arff'
        url3_test = 'data/epilepsy/EpilepsyDimension3_TEST.arff'

        x_dim1_test, target_test = get_data(url1_test)
        x_dim2_test, __ = get_data(url2_test)
        x_dim3_test, __ = get_data(url3_test)

        x_final_test = np.dstack([x_dim1_test, x_dim2_test, x_dim3_test])
        y_final_test, index_test = get_target(target_test, normal_class)

        test_set = TensorDataset(torch.Tensor(x_final_test), torch.Tensor(y_final_test), torch.Tensor(index_test))
        self.test_set = test_set

def get_data(url):
    """
    input: path to arff data file
    This function loads the arff file, then converts into dataframe.
    The dataframe is then split into x and y.
    output: x is dataframe object without the last column. y is series.
    """
    loaded = arff.loadarff(url)
    df = pd.DataFrame(loaded[0])
    
    # dropping the last column of dataframe
    # it is still a dataframe object
    x = df.iloc[:, :-1].to_numpy()

    # getting last column as series, not dataframe object
    # as dataframe object is using iloc[:, -1:]
    y = df.iloc[:, -1]

    return x, y


def get_target(y, normal_class):
    """
    input: pandas series. last column of dataframe.
    This function converts the byte string of series and compare to each classification group
    Each class is represented as a number.
    output: returns numpy array of numbers and index array
    """
    y_new = []
    y_temp = []
    idx = []
    length = len(y)

    for i in range(0, length):
        if y[i].decode('UTF-8') == 'EPILEPSY':
            y_temp.append(0)
        elif y[i].decode('UTF-8') == 'SAWING':
            y_temp.append(1)
        elif y[i].decode('UTF-8') == 'RUNNING':
            y_temp.append(2)
        elif y[i].decode('UTF-8') == 'WALKING':
            y_temp.append(3)
        idx.append(i)

    for i in range(0, length):
        if y_temp[i] == normal_class:
            y_new.append(1) # normal
        else:
            y_new.append(0) # anomaly

    return np.array(y_new), np.array(idx)

def get_training_set(x, y, idx):
    """
    Input: x, y, index of training set from data file
    This function only collects the normal data from train set.
    The model only trains on normal data of the train set.
    Output: x, y, index of normal data only in train set.
    """
    x_final = []
    y_final = []
    idx_final = []

    for i in range(0, len(x)):
        if y[i] == 1:
            x_final.append(x[i])
            y_final.append(y[i])
    
    for i in range(0, len(x_final)):
        idx_final.append(i)
    
    return np.array(x_final), np.array(y_final), np.array(idx_final)
