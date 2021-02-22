import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
# input is in the size of 120x8


class GaitSessionDataset(Dataset):
    def __init__(self, file_path, data_length=100, dimension=6):
        self.data_length = data_length
        self.dimension = dimension
        self.file_path = file_path
        number_of_feature = self.dimension*self.data_length

        xy = np.genfromtxt(self.file_path, delimiter=' ', dtype=np.float32)

        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:number_of_feature])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # data_idx = np.reshape(self.x_data[idx, :], (self.data_length,  self.dimension), order='A')
        sample = {'data': self.x_data[idx], 'label': self.y_data[idx]}
        return sample

    def getDataFile(self):
        return self.file_path


class GaitSessionTriplet(Dataset):
    def __init__(self, file_path, data_length=100, dimension=6, train=True, transform=None):

        self.data_length = data_length
        self.dimension = dimension
        self.file_path = file_path
        number_of_feature = self.dimension * self.data_length

        xy = np.genfromtxt(self.file_path, delimiter=' ', dtype=np.float32)

        self.len = xy.shape[0]
        # self.x_data = torch.from_numpy(xy[:, 0:number_of_feature])
        # self.y_data = torch.from_numpy(xy[:, [-1]])
        self.x_data = xy[:, 0:number_of_feature]
        self.y_data = xy[:, -1]

        self.index = np.arange(len(self.y_data))

        self.is_train = train
        # self.transform = transform
        # self.to_pil = transforms.ToPILImage()

        if self.is_train:
            self.index = np.arange(len(self.y_data))
            # self.images = df.iloc[:, 1:].values.astype(np.uint8)
            # self.labels = df.iloc[:, 0].values
            # self.index = df.index.values
        # else:
            # self.images = df.values.astype(np.uint8)

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        if self.is_train:
            # get the anchor item
            anchor_label = self.y_data[idx]
            anchor_segment = self.x_data[idx]

            # get the positive item
            posititve_list = self.index[self.index != idx][self.y_data[self.index != idx] == anchor_label]
            # positive_list = self.index[self.index != item][self.labels[self.index != item] == anchor_label]
            if(posititve_list.size==0):
                positive_item = idx
            else:
                positive_item = random.choice(posititve_list)
            positive_segment = self.x_data[positive_item]

            # get the negative item
            negative_list = self.index[self.index != idx][self.y_data[self.index != idx] != anchor_label]
            negative_item = random.choice(negative_list)
            negative_segment = self.x_data[negative_item]

            return anchor_segment, positive_segment, negative_segment, anchor_label
            return anchor_img, positive_img, negative_img, anchor_label

        else:
            sample = {'data': self.x_data[idx], 'label': self.y_data[idx]}
            return sample


#=========================================================================================================
class GaitCycleDataset1(Dataset):
    def __init__(self, file_path, data_length, dimension):
        self.data_length = data_length
        self.dimension = dimension
        self.file_path = file_path

        all_data = np.genfromtxt(self.file_path, delimiter=' ', dtype=np.float64)
        # all_data = np.loadtxt(self.file_path, delimiter=',', dtype=np.float64)
        np.set_printoptions(precision=16)

        self.x_data = all_data[:, 0:-1]

        self.y_data = all_data[:, -1]

        self.len = all_data.shape[0]
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data_idx = np.reshape(self.x_data[idx, :], (self.data_length,  self.dimension), order='F')
        sample = {'data': data_idx, 'label': self.y_data[idx]}
        return sample

    def getDataFile(self):
        return self.file_path


class GaitCycleDataset(Dataset):
    def __init__(self, file_path, data_length, dimension):
        self.data_length = data_length
        self.dimension = dimension
        self.file_path = file_path
        number_of_feature = self.dimension*self.data_length

        xy = np.genfromtxt(self.file_path, delimiter=' ', dtype=np.float32)

        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:number_of_feature])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # data_idx = np.reshape(self.x_data[idx, :], (self.data_length,  self.dimension), order='A')
        sample = {'data': self.x_data[idx], 'label': self.y_data[idx]}
        return sample

    def getDataFile(self):
        return self.file_path