import os
import glob
import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class oasis_pkl_loader(Dataset): # oasis_pkl_loader

    def __init__(self,
            root_dir = './../../../data/oasis_pkl/',
            split = 'train', # train, val or test
        ):
        self.root_dir = root_dir
        self.split = split

        train_dir = os.path.join(root_dir, 'train/')
        val_dir = os.path.join(root_dir, 'val/')

        if self.split == 'train':
            self.total_list = glob.glob(train_dir + '*.pkl')
        elif self.split == 'val' or self.split == 'test':
            self.total_list = glob.glob(val_dir + '*.pkl')
        else:
            raise ValueError('Invalid split name')

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):

        fp = self.total_list[idx]

        if self.split == 'train':
            tar_list = self.total_list.copy()
            tar_list.remove(fp)
            random.shuffle(tar_list)
            tar_file = tar_list[0]
            x, x_seg = pkload(fp)
            y, y_seg = pkload(tar_file)
        elif self.split == 'val' or self.split == 'test':
            x, y, x_seg, y_seg = pkload(fp)

        x, x_seg = x[None, ...], x_seg[None, ...]
        y, y_seg = y[None, ...], y_seg[None, ...]

        x, x_seg = np.ascontiguousarray(x), np.ascontiguousarray(x_seg)
        y, y_seg = np.ascontiguousarray(y), np.ascontiguousarray(y_seg)

        x, x_seg = torch.from_numpy(x), torch.from_numpy(x_seg)
        y, y_seg = torch.from_numpy(y), torch.from_numpy(y_seg)

        return x, x_seg, y, y_seg, idx