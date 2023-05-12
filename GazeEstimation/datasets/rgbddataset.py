import os
import pickle
from logging import getLogger

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm

logger = getLogger('logger')

ACTIVITES = ['standing', 'walking', 'sitting', 'lying']


def load_dataset(dataset_dir, config):
    with open(os.path.join(dataset_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    data_config = config['DATA']
    if 'TEST_PID' in data_config.keys():
        test_pids = data_config['TEST_PID']
    else:
        raise KeyError('No key of `TEST_PID` in config["DATA"]')

    if 'VAL_PID' in data_config.keys():
        val_pids = data_config['VAL_PID']
    else:
        val_pids = None

    if 'TRAIN_PID' in data_config.keys():
        train_pids = data_config['TRAIN_PID']
    else:
        train_pids = None

    logger.info(f'train: {train_pids}, val: {val_pids}, test: {test_pids}')
    train, val, test = _load_indices(metadata, train_pids, val_pids, test_pids)
    feature_type = data_config['FEATURE_TYPE']
    train = Dataset(dataset_dir, metadata, train, feature_type)
    val = Dataset(dataset_dir, metadata, val, feature_type)
    test = Dataset(dataset_dir, metadata, test, feature_type)
    return train, val, test

def _load_indices(meta_data, train_pid, val_pid, test_pid):
    train, val, test = [], [], []
    for i, pid in enumerate(meta_data['pid']):
        if pid == test_pid:
            test.append(i)
        elif pid == val_pid:
            val.append(i)
        elif pid in train_pid:
            train.append(i)

    logger.info(f'{len(train)=}, {len(val)=}, {len(test)=}')
    return train, val, test


class Dataset(data.Dataset):

    def __init__(self, dataset_dir, metadata, indices, feature_type):

        self.dataset_dir = dataset_dir
        self.metadata = metadata
        self.indices = indices
        self.feature_type = feature_type

    @staticmethod
    def normalize(tensor):
        tensor = tensor.div(255)
        dtype = tensor.dtype
        mean = torch.as_tensor([0.5 for _ in range(tensor.shape[0])],
                               dtype=dtype,
                               device=tensor.device).view(-1, 1, 1)
        std = torch.as_tensor([0.5 for _ in range(tensor.shape[0])],
                              dtype=dtype,
                              device=tensor.device).view(-1, 1, 1)
        return tensor.sub_(mean).div_(std)

    @staticmethod
    def load_tensor(path, dtype):
        return torch.load(path)

    def __getitem__(self, index):
        index = self.indices[index]
        tensor_p = os.path.join(
                        self.dataset_dir,
                        self.metadata['pid'][index],
                        'tensor',
                        self.metadata['activity'][index],
                        f'{self.metadata["frameIndex"][index]}.pt',
                   )

        tensor = torch.load(tensor_p)
        if self.feature_type == 'rgbd':
            pass
        elif self.feature_type == 'rgb':
            tensor = tensor[:3, :, :]
        else:
            raise TypeError('Unexpected feature type: {self.feature_type=}')
        tensor = self.normalize(tensor)
        gaze = np.array([self.metadata['labelDotX'][index],
                         self.metadata['labelDotY'][index]], np.float32)
        gaze = torch.FloatTensor(gaze)
        return tensor, gaze

    def __len__(self):
        return len(self.indices)
