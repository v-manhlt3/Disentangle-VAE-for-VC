import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck=64, latent_dim=256):
        super(Encoder, self).__init__()
        self.dim_neck = dim_neck
        # self.freq = freq
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80 if i==0 else 512,
                         512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)
        self.latent_code = LinearNorm(8192, latent_dim)

    def forward(self, x):
        shape = x.shape
        # x = x.transpose(-1,-2)
        # c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        # x = torch.cat((x, c_org), dim=1)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        # out_forward = outputs[:, :, :self.dim_neck]
        # out_backward = outputs[:, :, self.dim_neck:]
        outputs = outputs.reshape(shape[0], -1)
        codes = self.latent_code(outputs)
        
        # codes = []
        # for i in range(0, outputs.size(1), self.freq):
        #     codes.append(torch.cat((out_forward[:,i+self.freq-1,:],out_backward[:,i,:]), dim=-1))

        return codes

####### Encoder with no speaker identity is fed into input###########################################

def get_zeros_tensor(batch, dimensions):
    return torch.zeros(batch, dimensions)
      
########################################################################################################
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_emb, dim_pre, latent_dim=256):
        super(Decoder, self).__init__()
        self.dec_linear = LinearNorm(latent_dim, 8192)
        self.lstm1 = nn.LSTM(dim_neck*2, dim_pre, 1, batch_first=True)
        self.dim_neck = dim_neck
        self.dim_emb = dim_emb
        self.dim_pre = dim_pre
        self.latent_dim = latent_dim
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        
        self.linear_projection = LinearNorm(1024, 80)

    def forward(self, x):
        x = self.dec_linear(x)
        x = x.view(x.shape[0],-1, self.dim_neck*2)
        #self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        outputs, _ = self.lstm2(x)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(512,
                             512,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(512))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(80))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    

###################################################################################################

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck=64, dim_emb=256, dim_pre=512):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(dim_neck, dim_emb)
        self.decoder = Decoder(dim_neck, dim_emb, dim_pre)
        self.postnet = Postnet()

    def forward(self, x):
                
        # codes = self.encoder(x, c_org)
        # if c_trg is None:
        #     return torch.cat(codes, dim=-1)
        
        # tmp = []
        # for code in codes:
        #     tmp.append(code.unsqueeze(1).expand(-1,int(x.size(1)/len(codes)),-1))
        # code_exp = torch.cat(tmp, dim=1)
        # print('code_exp shape: ', code_exp.shape)
        # print('c_trg shape: ', c_trg.unsqueeze(1).expand(-1,x.size(1),-1).shape)
        # print('max value: ', torch.max(c_org[0]))

        encoder_outputs = self.encoder(x)
        
        mel_outputs = self.decoder(encoder_outputs)
                
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)
        
        return mel_outputs, mel_outputs_postnet


data = torch.randn(10,80, 64)
model = Generator()
output,_ = model(data)

print(data.shape)