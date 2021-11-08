import pandas as pd
import numpy as np
import torch

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

class NATOPS_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class):

        super().__init__(root)
        self.n_classes = 2
        self.normal_class = normal_class

        print("normal class: ", normal_class)

        # train set
        #load data file path
        url1_train = 'data/natops/NATOPSDimension1_TRAIN.arff'
        url2_train = 'data/natops/NATOPSDimension2_TRAIN.arff'
        url3_train = 'data/natops/NATOPSDimension3_TRAIN.arff'
        url4_train = 'data/natops/NATOPSDimension4_TRAIN.arff'
        url5_train = 'data/natops/NATOPSDimension5_TRAIN.arff'
        url6_train = 'data/natops/NATOPSDimension6_TRAIN.arff'
        url7_train = 'data/natops/NATOPSDimension7_TRAIN.arff'
        url8_train = 'data/natops/NATOPSDimension8_TRAIN.arff'
        url9_train = 'data/natops/NATOPSDimension9_TRAIN.arff'
        url10_train = 'data/natops/NATOPSDimension10_TRAIN.arff'
        url11_train = 'data/natops/NATOPSDimension11_TRAIN.arff'
        url12_train = 'data/natops/NATOPSDimension12_TRAIN.arff'
        url13_train = 'data/natops/NATOPSDimension13_TRAIN.arff'
        url14_train = 'data/natops/NATOPSDimension14_TRAIN.arff'
        url15_train = 'data/natops/NATOPSDimension15_TRAIN.arff'
        url16_train = 'data/natops/NATOPSDimension16_TRAIN.arff'
        url17_train = 'data/natops/NATOPSDimension17_TRAIN.arff'
        url18_train = 'data/natops/NATOPSDimension18_TRAIN.arff'
        url19_train = 'data/natops/NATOPSDimension19_TRAIN.arff'
        url20_train = 'data/natops/NATOPSDimension20_TRAIN.arff'
        url21_train = 'data/natops/NATOPSDimension21_TRAIN.arff'
        url22_train = 'data/natops/NATOPSDimension22_TRAIN.arff'
        url23_train = 'data/natops/NATOPSDimension23_TRAIN.arff'
        url24_train = 'data/natops/NATOPSDimension24_TRAIN.arff'

        # get x and y as dataframe
        x_dim1_train, target_train = get_data(url1_train)
        x_dim2_train, __ = get_data(url2_train)
        x_dim3_train, __ = get_data(url3_train)
        x_dim4_train, __ = get_data(url4_train)
        x_dim5_train, __ = get_data(url5_train)
        x_dim6_train, __ = get_data(url6_train)
        x_dim7_train, __ = get_data(url7_train)
        x_dim8_train, __ = get_data(url8_train)
        x_dim9_train, __ = get_data(url9_train)
        x_dim10_train, __ = get_data(url10_train)
        x_dim11_train, __ = get_data(url11_train)
        x_dim12_train, __ = get_data(url12_train)
        x_dim13_train, __ = get_data(url13_train)
        x_dim14_train, __ = get_data(url14_train)
        x_dim15_train, __ = get_data(url15_train)
        x_dim16_train, __ = get_data(url16_train)
        x_dim17_train, __ = get_data(url17_train)
        x_dim18_train, __ = get_data(url18_train)
        x_dim19_train, __ = get_data(url19_train)
        x_dim20_train, __ = get_data(url20_train)
        x_dim21_train, __ = get_data(url21_train)
        x_dim22_train, __ = get_data(url22_train)
        x_dim23_train, __ = get_data(url23_train)
        x_dim24_train, __ = get_data(url24_train)

        # combine 24 dimensions of x
        x_train = np.dstack([x_dim1_train, x_dim2_train, x_dim3_train, x_dim4_train, x_dim5_train, x_dim6_train, x_dim7_train, x_dim8_train, x_dim9_train, x_dim10_train, x_dim11_train, x_dim12_train, x_dim13_train, x_dim14_train, x_dim15_train, x_dim16_train, x_dim17_train, x_dim18_train, x_dim19_train, x_dim20_train, x_dim21_train, x_dim22_train, x_dim23_train, x_dim24_train])
        # process output y and produce index
        y_train, index_train = get_target(target_train, normal_class)

        # train only on normal data, extracting normal data
        x_final_train, y_final_train, index_final_train = get_training_set(x_train, y_train, index_train)

        # print("size: ", x_final_train.shape)
        train_set = TensorDataset(torch.Tensor(x_final_train), torch.Tensor(y_final_train), torch.Tensor(index_final_train))
        self.train_set = train_set

        # set up testing set
        url1_test = 'data/natops/NATOPSDimension1_TEST.arff'
        url2_test = 'data/natops/NATOPSDimension2_TEST.arff'
        url3_test = 'data/natops/NATOPSDimension3_TEST.arff'
        url4_test = 'data/natops/NATOPSDimension4_TEST.arff'
        url5_test = 'data/natops/NATOPSDimension5_TEST.arff'
        url6_test = 'data/natops/NATOPSDimension6_TEST.arff'
        url7_test = 'data/natops/NATOPSDimension7_TEST.arff'
        url8_test = 'data/natops/NATOPSDimension8_TEST.arff'
        url9_test = 'data/natops/NATOPSDimension9_TEST.arff'
        url10_test = 'data/natops/NATOPSDimension10_TEST.arff'
        url11_test = 'data/natops/NATOPSDimension11_TEST.arff'
        url12_test = 'data/natops/NATOPSDimension12_TEST.arff'
        url13_test = 'data/natops/NATOPSDimension13_TEST.arff'
        url14_test = 'data/natops/NATOPSDimension14_TEST.arff'
        url15_test = 'data/natops/NATOPSDimension15_TEST.arff'
        url16_test = 'data/natops/NATOPSDimension16_TEST.arff'
        url17_test = 'data/natops/NATOPSDimension17_TEST.arff'
        url18_test = 'data/natops/NATOPSDimension18_TEST.arff'
        url19_test = 'data/natops/NATOPSDimension19_TEST.arff'
        url20_test = 'data/natops/NATOPSDimension20_TEST.arff'
        url21_test = 'data/natops/NATOPSDimension21_TEST.arff'
        url22_test = 'data/natops/NATOPSDimension22_TEST.arff'
        url23_test = 'data/natops/NATOPSDimension23_TEST.arff'
        url24_test = 'data/natops/NATOPSDimension24_TEST.arff'

        x_dim1_test, target_test = get_data(url1_test)
        x_dim2_test, __ = get_data(url2_test)
        x_dim3_test, __ = get_data(url3_test)
        x_dim4_test, __ = get_data(url4_test)
        x_dim5_test, __ = get_data(url5_test)
        x_dim6_test, __ = get_data(url6_test)
        x_dim7_test, __ = get_data(url7_test)
        x_dim8_test, __ = get_data(url8_test)
        x_dim9_test, __ = get_data(url9_test)
        x_dim10_test, __ = get_data(url10_test)
        x_dim11_test, __ = get_data(url11_test)
        x_dim12_test, __ = get_data(url12_test)
        x_dim13_test, __ = get_data(url13_test)
        x_dim14_test, __ = get_data(url14_test)
        x_dim15_test, __ = get_data(url15_test)
        x_dim16_test, __ = get_data(url16_test)
        x_dim17_test, __ = get_data(url17_test)
        x_dim18_test, __ = get_data(url18_test)
        x_dim19_test, __ = get_data(url19_test)
        x_dim20_test, __ = get_data(url20_test)
        x_dim21_test, __ = get_data(url21_test)
        x_dim22_test, __ = get_data(url22_test)
        x_dim23_test, __ = get_data(url23_test)
        x_dim24_test, __ = get_data(url24_test)

        x_final_test = np.dstack([x_dim1_test, x_dim2_test, x_dim3_test, x_dim4_test, x_dim5_test, x_dim6_test, x_dim7_test, x_dim8_test, x_dim9_test, x_dim10_test, x_dim11_test, x_dim12_test, x_dim13_test, x_dim14_test, x_dim15_test, x_dim16_test, x_dim17_test, x_dim18_test, x_dim19_test, x_dim20_test, x_dim21_test, x_dim22_test, x_dim23_test, x_dim24_test])
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
        if y[i].decode('UTF-8') == '1.0':
            y_temp.append(0)
        elif y[i].decode('UTF-8') == '2.0':
            y_temp.append(1)
        elif y[i].decode('UTF-8') == '3.0':
            y_temp.append(2)
        elif y[i].decode('UTF-8') == '4.0':
            y_temp.append(3)
        elif y[i].decode('UTF-8') == '5.0':
            y_temp.append(4)
        elif y[i].decode('UTF-8') == '6.0':
            y_temp.append(5)
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