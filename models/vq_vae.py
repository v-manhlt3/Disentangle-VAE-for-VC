import torch
import torch.nn as nn
import torch.nn.Functional as F
from torch.autograd import Varibale
from models.vq_vae_block import *


class Encoder(nn.Module):

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
                use_kaiming_normal, input_features_type, features_filters, sampling_rate, device, verbose=False):
        
        super().__init__()
        
        self._conv1 = Conv1DBuilder.build(
            in_channels=features_filters,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self._conv2 = Conv1DBuilder.build(
            inp_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            using_kaiming_normal=use_kaiming_normal,
            padding=1
        )
        self._conv3 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            use_kaiming_normal=use_kaiming_normal,
            padding=2
        )
        self._conv4 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )
        self._conv5 = Conv1DBuilder(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal
        )

        self._input_feature_type = input_features_type
        self._features_filters = features_filters
        self._sampling_rate = sampling_rate
        self._device = device
        self._verbose = verbose

    def forward(self, inputs):
        
        x_conv1 = F.relu(self._conv1(inputs))

        x_conv2 = F.relu(self._conv2(x_conv1))
        x_conv3 = F.relu(self._conv3(x_conv2))
        x_conv4 = F.relu(self._conv4(x_conv3))
        x_conv5 = F.relu(self._conv5(x_conv4))

        x_residual = self._residual_stack(x_conv5) + x_conv5 

        return x_residual

class Decoder(nn.Module):

    def __init__(self,)
    