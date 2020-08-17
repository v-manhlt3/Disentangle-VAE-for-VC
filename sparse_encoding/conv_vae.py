import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from torch.nn import functional as F 
from sparse_encoding.variational_base_model import VariationalBaseModel

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

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


class ConvVSC(nn.Module):

    def __init__(self, input_sz = (1, 64, 80),
                kernel_szs = [512,512,512],
                hidden_sz: int = 256,
                latent_sz: int = 32,
                c: float = 512,
                c_delta: float = 0.001,
                beta: float = 0.1,
                beta_delta: float = 0,
                dim_neck=64, latent_dim=256, dim_pre=512):
        super(ConvVSC, self).__init__()
        self._input_sz = input_sz
        self._channel_szs = [input_sz[0]] + kernel_szs
        self._hidden_sz = hidden_sz
        self._c = c
        self._c_delta = c_delta
        self._beta = beta
        self._beta_delta = beta_delta
        self._latent_dim = latent_dim
        self.dim_neck = dim_neck
        self.postnet = Postnet()

        ############################## Encoder Architecture ###################
        self.enc_modules = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(80 if i==0 else 512,
                        512,
                        kernel_size=5, stride=1,
                        padding=2,
                        dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512)
            )
            self.enc_modules.append(conv_layer)
        self.enc_modules = nn.ModuleList(self.enc_modules)
        self.enc_lstm = nn.LSTM(dim_pre, dim_neck, 2, batch_first=True, bidirectional=True)
        self.mu = LinearNorm(8192, latent_dim)

        self.logvar = LinearNorm(8192, latent_dim)
        self.logspike = LinearNorm(8192, latent_dim)

        ############################ Decoder Architecture ####################
        self.dec_linear = nn.Linear(latent_dim, 8192)
        self.dec_lstm1 = nn.LSTM(dim_neck*2, 512, 1, batch_first=True)
        self.dec_modules = []

        for i in range(3):
            dec_conv_layer =  nn.Sequential(
                ConvNorm(dim_pre,
                        dim_pre,
                        kernel_size=5, stride=1,
                        padding=2, dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre)
            )
            self.dec_modules.append(dec_conv_layer)
        self.dec_modules = nn.ModuleList(self.dec_modules)

        self.dec_lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        self.dec_linear2 = LinearNorm(1024, 80)
        self.apply(init_weights)
    

    def encode(self, x):
        shape = x.shape
        
        # x = x.view(shape[0], shape[2], shape[3])
        # print('input data shape: ', x.shape)
        for layer in self.enc_modules:
            x = F.relu(layer(x))
        
        x = x.transpose(1, 2)

        self.enc_lstm.flatten_parameters()

        outputs, _ = self.enc_lstm(x)
        # outs_forward = outputs[:,:,:self.dim_neck]
        # outs_backward = outputs[:,:,self.dim_neck:]
        # print('---------------------encoders output: ', outputs.shape)
        outputs = outputs.reshape(shape[0], -1)

        mu = self.mu(outputs)
        logvar = self.logvar(outputs)
        logspike = -F.relu(-self.logspike(outputs))

        return mu, logvar, logspike

    def reparameterize(self, mu, logvar, logspike):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        gaussian =  eps.mul(std).add_(mu)
        eta = torch.randn_like(std)
        selection = F.sigmoid(self._c*(eta +  logspike.exp()-1))

        return selection.mul(gaussian)

    def decode(self, z):
        # print('latent size: ',z.shape)
        output = self.dec_linear(z)
        output = output.view(z.shape[0],-1, self.dim_neck*2)
        
        # print('-------------- decoder input shape: ', output.shape)
        output,_ = self.dec_lstm1(output)
        
        output = output.transpose(-1, -2)
        # print('-------------- output lstm shape: ', output.shape)
        for layer in self.dec_modules:
            output = F.relu(layer(output))
        output = output.transpose(-1, -2)
        # print('-------------- output lstm shape2: ', output.shape)
        output,_ = self.dec_lstm2(output)
        output = F.relu(self.dec_linear2(output))
        return output.transpose(-1, -2)

    def forward(self, x, train=True):
        mu, logvar, logspike = self.encode(x)
        if train:
            z =  self.reparameterize(mu, logvar, logspike)
        else:
            z = mu
        
        x_hat0 = self.decode(z)
        x_hat = x_hat0 + self.postnet(x_hat0)
        return x_hat0, x_hat, mu, logvar, logspike

    def update_c(self):
        self._c += self._c_delta
    
    def update_beta(self):
        self._beta += self._beta_delta

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


class ConvolutionalVSC(VariationalBaseModel):
    def __init__(self, dataset, width, height,
                latent_sz, learning_rate, alpha,
                log_interval, normalize, batch_size, channels=1, device=torch.device("cuda"),
                latent_dim=256, beta=0.1):
        super().__init__(dataset, width, height, channels,latent_sz, learning_rate,
                        device, log_interval, batch_size)
        
        self.alpha = alpha
        self.lr= learning_rate
        self.latent_dim = latent_dim
        # self.hidden_sz = hidden_sz
        # self.kernel_szs = [int(ks) for ks in str(kernel_szs).split(',')]

        self.model = ConvVSC(latent_dim=self.latent_dim, beta=0.1).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_losses = []
        self.test_losses = []

    # Reconstruction + KL divergence loss sumed over all elements of batch
    def loss_function(self, x, x_recon0, x_recon, mu, logvar, logspike, train=False):
        
        shape = x.shape
        # print('-------------------shape of mel: ', x.shape)
        # print('-------------------shape of recons mel: ', x_recon.shape)
        # MSE0 = (shape[-1]*shape[-2])*torch.nn.functional.l1_loss(x, x_recon0)
        # MSE = (shape[-1]*shape[-2])*torch.nn.functional.l1_loss(x, x_recon)
        MSE0 = torch.nn.functional.l1_loss(x, x_recon0, reduction='sum')
        MSE = torch.nn.functional.l1_loss(x, x_recon, reduction='sum')

        spike= torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6)

        prior1 = -0.5 * torch.sum(spike.mul(1 + logvar - mu.pow(2) - logvar.exp()))
        prior21 = (1 - spike).mul(torch.log((1 - spike) / (1 - self.alpha)))
        prior22 = spike.mul(torch.log(spike /  self.alpha))
        prior2 = torch.sum(prior21 + prior22)
        PRIOR = torch.mean(prior1 + prior2)

        LOSS = MSE0 + MSE + self.model._beta*PRIOR

        log = {
            'LOSS': LOSS.item(),
            'MSE': MSE.item(),
            'PRIOR': PRIOR.item(),
            'prior1': prior1.item(),
            'prior2': prior2.item()
        }
        if train:
            self.train_losses.append(log)
        else:
            self.test_losses.append(log)
        
        return LOSS, MSE0,MSE, PRIOR

    def update_(self):
        self.model.update_c()
        self.model.update_beta()


# model = ConvVSC()
# model = model.cuda()
# data = torch.randn(10, 64, 80).cuda()

# output = model(data)
# print(output[0].shape)




