import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 
import soundfile as sf
# def weight_init():

from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('Elapsed time function: {}'.format(end-start))
        return result
    return wrapper


class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, sefl).__init__()

        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class ConvNorm(nn.Module):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=None,
                dilation=1,
                bias=True,
                w_init_gain='linear'):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert(kernel_size%2==1)
            padding = int(dilation*(kernel_size-1)/2)

        self.conv = nn.Conv1d(in_channels, out_channels,
                             kernel_size=kernel_size, stride=stride,
                             padding=padding, dilation=dilation, bias=bias)

        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        conv_signal = self.conv(x)
        return conv_signal

class Encoder(nn.Module):
    def __init__(self, in_dim, kernel_sizes, strides, channels, latent_dim):
        super(Encoder, self).__init__()
        convolutions = []
        in_channels = in_dim
        assert(len(kernel_sizes)==len(strides)==len(channels))
        self.latent_dim = latent_dim

        for i in range(len(kernel_sizes)):
            conv_layer = nn.Conv1d(in_channels,
                                  channels[i],
                                  kernel_size=kernel_sizes[i],stride=strides[i],
                                  padding=(strides[i]//2),bias=True)
            batch_norm = nn.BatchNorm1d(channels[i])
            activation = nn.ReLU()

            nn.init.xavier_uniform_(conv_layer.weight)
            nn.init.zeros_(conv_layer.bias)

            convolutions.append(conv_layer)
            convolutions.append(batch_norm)
            convolutions.append(activation)
            in_channels = channels[i]

        self.convolutions = nn.ModuleList(convolutions)
        self.style = nn.Linear(200, latent_dim*2, bias=True)
        self.content = nn.Linear(200, latent_dim*2, bias=True)

        nn.init.xavier_uniform_(self.style.weight)
        nn.init.zeros_(self.style.bias)
        nn.init.xavier_uniform_(self.content.weight)
        nn.init.zeros_(self.content.bias)

        

    def forward(self, x):
        """
        input size x (BxLxC)
        output latent_dim z
        """
        shape = x.shape
        # print(shape)
        x = x.unsqueeze(1)
        out = x
        for layer in self.convolutions:
            out = layer(out)
        # out = self.convolutions(x)
        out = out.view(shape[0], -1)

        style = self.style(out)
        content = self.content(out)
        style_mu = style[:,:self.latent_dim]
        style_logvar = style[:,self.latent_dim:]
        content_mu = content[:,self.latent_dim:]
        content_logvar = content[:,self.latent_dim:]
        
        return style_mu, style_logvar, content_mu, content_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, channels, kernel_sizes, strides, in_dim):
        super(Decoder, self).__init__()

        assert(len(channels)==len(kernel_sizes)==len(strides))
        in_channel = in_dim
        self.in_dim = in_dim
        self.latent_dim = latent_dim

        transpose_convs = []

        for i in range(len(kernel_sizes)):
            trans_conv = nn.ConvTranspose1d(in_channel, channels[i],
                                            kernel_size=kernel_sizes[i], stride=strides[i],
                                            padding=1,bias=True)
            batch_norm = nn.BatchNorm1d(channels[i])
            activation = nn.LeakyReLU()

            nn.init.xavier_uniform_(trans_conv.weight)
            nn.init.zeros_(trans_conv.bias)

            transpose_convs.append(trans_conv)
            transpose_convs.append(batch_norm)
            transpose_convs.append(activation)
            in_channel = channels[i]

        self.transpose_convs = nn.ModuleList(transpose_convs)
        self.linear_layer = nn.Linear(latent_dim*2, 200, bias=True)
        self.out_transpose_conv = nn.ConvTranspose1d(8, 1, kernel_size=1, stride=1, bias=True)
        self.tanh_layer = nn.Tanh()

        nn.init.xavier_uniform_(self.linear_layer.weight)
        nn.init.zeros_(self.linear_layer.bias)
        nn.init.xavier_uniform_(self.out_transpose_conv.weight)
        nn.init.zeros_(self.out_transpose_conv.bias)
    def reparameterization(self, mu, logvar, train=True):
        if train:
            eps = torch.autograd.Variable(torch.empty(logvar.shape).normal_())
            std = logvar.mul(0.5).exp_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    def forward(self, style_mu, style_logvar, content_mu, content_logvar):
        # shape = z.shape
        shape = style_mu.shape
        style = self.reparameterization(style_mu, style_logvar)
        content = self.reparameterization(content_mu, content_logvar)
        z = torch.cat((style,content), dim=-1)

        out = self.linear_layer(z)
        out = out.view(shape[0], self.in_dim, -1)
        for layer in self.transpose_convs:
            out = layer(out)
        out = self.out_transpose_conv(out)
        out = self.tanh_layer(out)
        out = out.squeeze(1)
        return out[:,:16000]



if __name__=='__main__':

    # kernel_sizes = [10,10,10,10]
    # channels = [64,32,16,8]
    # strides = [5, 5, 5, 5]

    # data = torch.rand(4,16000)
    # encoder = Encoder(kernel_sizes=kernel_sizes, strides=strides, channels=channels, in_dim=1, latent_dim=28)
    # decoder = Decoder(kernel_sizes=kernel_sizes, strides=strides, channels=channels, in_dim=8, latent_dim=28)
    # mu1, logvar1, mu2, logvar2 = encoder(data)
    # print('mu1 shape: ',mu1.shape)
    # reconstructed_x = decoder(mu1, logvar1, mu2, logvar2)

    # # print('latent variabe: ', z)
    # # print('reconstructed x: ', reconstructed_x)
    # print('reconstructed shape', reconstructed_x.shape)
    import numpy as np
    from scipy.io.wavfile import write

    # data = np.random.uniform(-1,1,72000)
    # scale = np.int16(data/np.max(np.abs(data))*32767)
    # write('/home/manhlt/Desktop/test.wav', 72000, scale)
    data, samplerate = sf.read('/home/manhlt/Desktop/p225_001.wav')
    
    # scaled = np.int16()
    convert_data = ((data + 1)/2)*256
    min_val = np.min(convert_data)
    max_val = np.max(convert_data)
    print(min_val)
    print(max_val)
    # sample = np.random.choice(256, 32000).astype(np.int16)
    print(convert_data)
    print(data.size)
    # min_val = np.min(data)
    # print(min_val)
    sf.write('/home/manhlt/Desktop/sss.wav', convert_data, 16000)