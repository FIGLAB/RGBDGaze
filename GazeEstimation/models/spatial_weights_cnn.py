import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger('logger')


def init_conv(m, mean, variance, bias):
    nn.init.normal_(m.weight.data, mean, math.sqrt(variance))
    nn.init.constant_(m.bias.data, bias)
    return m


class SpatialWeightsCNN(nn.Module):

    def __init__(self, feature_type):
        super(SpatialWeightsCNN, self).__init__()
        self.feature_layer = nn.Sequential(models.alexnet(pretrained=True).features)
        if feature_type == 'rgbd':
            new_layer = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
            with torch.no_grad():
                new_layer.weight[:, :3, :, :] = self.feature_layer[0][0].weight
            logger.info(f'Reshape feature layer: {self.feature_layer[0][0].weight.shape} => {new_layer.weight.shape}')
            self.feature_layer[0][0] = new_layer
        elif feature_type == 'rgb':
            pass
        elif feature_type == 'd':
            new_layer = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
            logger.info(f'Reshape feature layer: {self.feature_layer[0][0].weight.shape} => {new_layer.weight.shape}')
            self.feature_layer[0][0] = new_layer
        else:
            raise TypeError(f'Unexpected feature type: {feature_type=}')

        self.conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)

        self.fc1 = nn.Linear(256 * 13**2, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)

        self._register_hook()
        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.normal_(self.conv1.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv2.weight, mean=0, std=0.01)
        nn.init.normal_(self.conv3.weight, mean=0, std=0.001)
        nn.init.constant_(self.conv1.bias, val=0.1)
        nn.init.constant_(self.conv2.bias, val=0.1)
        nn.init.constant_(self.conv3.bias, val=1)
        nn.init.normal_(self.fc1.weight, mean=0, std=0.005)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.0001)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.0001)
        nn.init.constant_(self.fc1.bias, val=1)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def _register_hook(self):
        n_channels = self.conv1.in_channels

        def hook(module, grad_in, grad_out):
            return tuple(grad / n_channels for grad in grad_in)

        self.handles = []
        self.handles.append(self.conv3.register_backward_hook(hook))

    def remove_hook(self):
        for handle in self.handles:
            handle.remove()

    def load_pretrained_data(self, state_dict):
        if state_dict['feature_layer.0.0.weight'].shape == self.feature_layer[0][0].weight.shape:
            self.load_state_dict(state_dict)
        else:
            first_layer_weight = state_dict['feature_layer.0.0.weight']
            tmp_layer = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
            self.feature_layer[0][0] = tmp_layer
            self.load_state_dict(state_dict)
            new_layer = nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2)
            with torch.no_grad():
                new_layer.weight[:, :3, :, :] = first_layer_weight
            self.feature_layer[0][0] = new_layer

    def forward(self, inputs):
        x = inputs
        x = self.feature_layer(x)
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        x = x * y
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=self.training)
        x = self.fc3(x)
        return x
