import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_weights_cnn import SpatialWeightsCNN

logger = logging.getLogger('logger')


def init_conv(m, mean, variance, bias):
    nn.init.normal_(m.weight.data, mean, math.sqrt(variance))
    nn.init.constant_(m.bias.data, bias)
    return m


class TwoStream(nn.Module):

    def __init__(self):
        super(TwoStream, self).__init__()
        self.rgb_spatial = SpatialWeightsCNN(feature_type='rgb')
        self.d_spatial = SpatialWeightsCNN(feature_type='d')

        self.fc1 = nn.Linear(256 * 13**2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 2)

        self._initialize_weight()

    def _initialize_weight(self):
        nn.init.normal_(self.fc1.weight, mean=0, std=0.005)
        nn.init.normal_(self.fc2.weight, mean=0, std=0.0001)
        nn.init.normal_(self.fc3.weight, mean=0, std=0.0001)
        nn.init.constant_(self.fc1.bias, val=1)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def load_pretrained_data(self, state_dict):

        try:
            self.load_state_dict(state_dict)
            logger.info('Load TwoStream from checkpoint')
        except BaseException:
            self.rgb_spatial.load_pretrained_data(state_dict)
            logger.info('Load RGB part of TwoStream from checkpoint')

    def forward(self, inputs):
        rgb = inputs[:, :3, :, :]
        d = inputs[:, 3, :, :].unsqueeze(1)

        rgb = self.rgb_spatial.feature_layer(rgb)
        rgb_y = F.relu(self.rgb_spatial.conv1(rgb))
        rgb_y = F.relu(self.rgb_spatial.conv2(rgb_y))
        rgb_y = F.relu(self.rgb_spatial.conv3(rgb_y))

        rgb = rgb * rgb_y
        rgb = rgb.view(rgb.size(0), -1)

        d = self.d_spatial.feature_layer(d)
        d_y = F.relu(self.d_spatial.conv1(d))
        d_y = F.relu(self.d_spatial.conv2(d_y))
        d_y = F.relu(self.d_spatial.conv3(d_y))
        d = d * d_y
        d = d.view(d.size(0), -1)

        x = torch.cat((rgb, d), 1)
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
        x = F.dropout(F.relu(self.fc2(x)), p=0.5, training=self.training)
        x = self.fc3(x)
        return x

    @torch.no_grad()
    def get_middle_tensors(self, inputs):
        rgb = inputs[:, :3, :, :]
        d = inputs[:, 3, :, :].unsqueeze(1)
        ret = {}

        rgb = self.rgb_spatial.feature_layer(rgb)
        ret['rgb'] = rgb
        rgb_y = F.relu(self.rgb_spatial.conv1(rgb))
        rgb_y = F.relu(self.rgb_spatial.conv2(rgb_y))
        rgb_y = F.relu(self.rgb_spatial.conv3(rgb_y))
        ret['rgb_y'] = rgb_y

        d = self.d_spatial.feature_layer(d)
        ret['d'] = d
        d_y = F.relu(self.d_spatial.conv1(d))
        d_y = F.relu(self.d_spatial.conv2(d_y))
        d_y = F.relu(self.d_spatial.conv3(d_y))
        ret['d_y'] = d_y
        return ret


if __name__ == '__main__':
    model = TwoStream()
    x = torch.rand([1, 4, 448, 448])
    print(model(x))
