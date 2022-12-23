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


def load_dataset(dataset_dir, config, activity=None):
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
    train, val, test = _load_indices(metadata, test_pids, train_pids, val_pids, activity)
    feature_type = data_config['FEATURE_TYPE']
    train = Dataset(dataset_dir, metadata, train, feature_type)
    val = Dataset(dataset_dir, metadata, val, feature_type)
    test = Dataset(dataset_dir, metadata, test, feature_type)
    return train, val, test

    train, val, test = [], [], []

    assert len(test_pids) != 0, 'test_pid should not be empty'
    remaining_pids = [p for p in set(metadata['pid']) if p not in test_pids]

    if train_pids is not None and val_pids is not None:
        pass
    elif train_pids is not None and val_pids is None:  # val_pids is None
        val_pids = [p for p in remaining_pids if p not in train_pids]
    elif train_pids is None and val_pids is not None:
        train_pids = [p for p in remaining_pids if p not in val_pids]
    else:   # both are None
        from sklearn.model_selection import train_test_split
        train_pids, val_pids = train_test_split(remaining_pids, test_size=0.11)

    logger.info(f'train: {train_pids} val: {val_pids}')

    # filter index by activity info
    for i, pid in enumerate(metadata['pid']):
        if activity is not None:
            if metadata['activity'][i] != activity:
                continue

        if pid in test_pids:
            test.append(i)

        if pid in val_pids:
            val.append(i)
        elif pid in train_pids:
            train.append(i)

    logger.info(f'Use {activity if activity is not None else "all"} activity')
    logger.info(f'Total number: {len(train)=}, {len(val)=}, {len(test)=}')
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
