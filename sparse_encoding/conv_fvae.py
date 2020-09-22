import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F 
from sparse_encoding.variational_base_fvae import VariationalBaseModel
import timeit
from sparse_encoding import utils
import torch.nn.init as init


torch.autograd.set_detect_anomaly(True)
def timer(function):
  def new_function(mu, logvar, is_cuda, batch_labels):
    start_time = timeit.default_timer()
    function(mu, logvar, is_cuda, batch_labels)
    elapsed = timeit.default_timer() - start_time
    print('Function "{name}" took {time} seconds to complete.'.format(name=function.__name__, time=elapsed))
  return new_function


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

### tile the output n times ##################################
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.cuda.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


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


class FVAE(nn.Module):

    def __init__(self, input_sz = (1, 64, 80),
                kernel_szs = [512,512,512],
                hidden_sz: int = 256,
                latent_sz: int = 32,
                c: float = 512,
                c_delta: float = 0.001,
                beta: float = 0.1,
                beta_delta: float = 0,
                dim_neck=64, latent_dim=256, dim_pre=512, batch_size=10):
        super(FVAE, self).__init__()

        self.batch_size = batch_size
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
                ConvNorm(36 if i==0 else 512,
                        512,
                        kernel_size=5, stride=2,
                        padding=2,
                        dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(512)
            )
            self.enc_modules.append(conv_layer)
        self.enc_modules = nn.ModuleList(self.enc_modules)
        self.enc_lstm = nn.LSTM(dim_pre, dim_neck, 2, batch_first=True, bidirectional=True)

        ################ mel-spectrogram#########################################
        # self.enc_linear = LinearNorm(8192, 512)

        # self.mu = LinearNorm(512, latent_dim) # length=64, 8192
        # self.logvar = LinearNorm(512, latent_dim)
        #########################################################################
        self.enc_linear = LinearNorm(4096, 512)

        self.mu = LinearNorm(512, latent_dim) # length=64, 8192
        self.logvar = LinearNorm(512, latent_dim)

        ############################ Decoder Architecture ####################
        self.dec_pre_linear1 = LinearNorm(latent_dim, 512)
        self.dec_pre_linear2 = LinearNorm(512, 4096)
        self.dec_lstm1 = nn.LSTM(dim_neck*2, 512, 1, batch_first=True)
        self.dec_modules = []

        for i in range(3):
            if i<2:
                dec_conv_layer =  nn.Sequential(
                    nn.ConvTranspose1d(dim_pre,
                            dim_pre,
                            kernel_size=5, stride=2,
                            padding=1, dilation=1),
                    nn.BatchNorm1d(dim_pre)
                )
                self.dec_modules.append(dec_conv_layer)
            else:
                dec_conv_layer =  nn.Sequential(
                    nn.ConvTranspose1d(dim_pre,
                            dim_pre,
                            kernel_size=5, stride=2,
                            padding=2, dilation=1),
                    nn.BatchNorm1d(dim_pre)
                )
                self.dec_modules.append(dec_conv_layer)
        self.dec_modules = nn.ModuleList(self.dec_modules)

        self.dec_lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        self.dec_linear2 = LinearNorm(1024, 36)
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
        outputs = outputs.reshape(shape[0], -1)
        outputs = self.enc_linear(outputs)

        mu = self.mu(outputs)
        logvar = self.logvar(outputs)

        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        epsilon = Variable(torch.empty(logvar.size()).normal_()).cuda()
        std = logvar.mul(0.5).exp_()
        return epsilon.mul(std).add_(mu)

    def decode(self, z):
        
        # print('latent size: ',z.shape)
        output = self.dec_pre_linear1(z)
        output = self.dec_pre_linear2(output)
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
        output = self.dec_linear2(output)
        return output.transpose(-1, -2)[:,:,:256]

    def forward(self, x, train=True):
        mu, logvar  = self.encode(x)
        # print('group style mu shape: ', group_style_mu.shape)
        if train:
            z =  self.reparameterize(mu, logvar)
        else:
            z = mu

        x_hat0 = self.decode(z)
        # x_hat = x_hat0 + self.postnet(x_hat0)
        x_hat = x_hat0
        return x_hat0, x_hat, mu, logvar, z

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

###################  Discriminator for latent space come from q(z) or q_bar(z) #####################################################
class Discriminator(nn.Module):
    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init

        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()
####################################################################################################################################

class ConvolutionalFVAE(VariationalBaseModel):
    def __init__(self, dataset, width, height,
                latent_sz, learning_rate, alpha,
                log_interval, normalize, batch_size, gamma,channels=1, device=torch.device("cuda"),
                latent_dim=256, beta=0.1):
        super().__init__(dataset, width, height, channels,latent_sz, learning_rate,
                        device, log_interval, batch_size)
        
        self.alpha = alpha
        self.lr= learning_rate
        self.latent_dim = latent_dim
        self.gamma = gamma
        # self.hidden_sz = hidden_sz
        # self.kernel_szs = [int(ks) for ks in str(kernel_szs).split(',')]

        self.model = FVAE(latent_dim=self.latent_dim, beta=0.1, batch_size=batch_size).to(device)
        self.D = Discriminator(latent_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=(1e-1*self.lr))
        self.train_losses = []
        self.test_losses = []

        # self.zeros = torch.zeros(batch_size, dtype=torch.long).cuda()
        # self.ones = torch.ones(batch_size, dtype=torch.long).cuda()

    # Reconstruction + KL divergence loss sumed over all elements of batch
    def loss_function(self, x, x_recon0, x_recon, mu, logvar, z, train=False):
        
        shape = x.shape
        D_z = self.D(z)

        MSE0 = torch.nn.functional.l1_loss(x, x_recon0, reduction='sum').div(self.batch_size)
        MSE = torch.nn.functional.l1_loss(x, x_recon, reduction='sum').div(self.batch_size)

        kl_loss = (-0.5)*torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).sum().div(self.batch_size)
        vae_tc_loss = (D_z[:,:1] - D_z[:,1:]).sum().div(self.batch_size)

        
        
        LOSS = 10*MSE0 + self.model._beta*kl_loss - self.gamma*vae_tc_loss
        # LOSS = 10*MSE0 + 10*MSE + self.model._beta*kl_loss - self.gamma*vae_tc_loss
        
        return LOSS, MSE0,MSE, kl_loss, vae_tc_loss

    def dis_loss_function(self, z, z_prime):

        D_z = self.D(z)
        z_permutation = utils.permute_dims(z_prime).detach()
        D_z_permu = self.D(z_permutation)
        zeros = torch.zeros(D_z.shape[0], dtype=torch.long).cuda()
        ones = torch.ones(D_z.shape[0], dtype=torch.long).cuda()
        dis_tc_loss = 0.5*(F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_permu, ones))

        return dis_tc_loss

    def update_(self):
        self.model.update_c()
        self.model.update_beta()


# model = ConvVSC()
# model = model.cuda()
# data = torch.randn(10, 64, 80).cuda()

# output = model(data)
# print(output[0].shape)




