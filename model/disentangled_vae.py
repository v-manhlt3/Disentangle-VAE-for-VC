import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F 
# from sparse_encoding.variational_base_mulvae_model import VariationalBaseModel
# from sparse_encoding.variational_base_acvae import VariationalBaseModelGVAE
# from variational_base_acvae import VariationalBaseModelGVAE
from model.variational_base_vae import VariationalBaseModelVAE
import timeit
# from sparse_encoding import utils
from torch.autograd import Variable


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

### tile the output n times ##################################
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.cuda.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

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
############################################################################

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


class DisentangledVAE(nn.Module):

    def __init__(self, speaker_size,input_sz = (1, 64, 80),
                kernel_szs = [512,512,512],
                hidden_sz: int = 256,
                latent_sz: int = 32,
                c: float = 512,
                c_delta: float = 0.001,
                beta: float = 0.1,
                beta_delta: float = 0,
                dim_neck=64, latent_dim=64, dim_pre=512, batch_size=10):
        super(DisentangledVAE, self).__init__()

        self.batch_size = batch_size
        self._input_sz = input_sz
        self._channel_szs = [input_sz[0]] + kernel_szs
        self._hidden_sz = hidden_sz
        self._c = c
        self._c_delta = c_delta
        self._beta = beta
        self._beta_delta = beta_delta
        self.latent_dim = latent_dim
        self.dim_neck = dim_neck
        self.speaker_size = speaker_size
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

        self.enc_linear = LinearNorm(8192, 2048)

        self.style = LinearNorm(2048, self.speaker_size*2)
        self.content = LinearNorm(2048, (latent_dim - self.speaker_size)*2)
        ############################ Decoder Architecture ####################
        self.dec_pre_linear1 = nn.Linear(latent_dim, 2048)
        self.dec_pre_linear2 = nn.Linear(2048, 8192)
        self.dec_lstm1 = nn.LSTM(dim_neck*2, 512, 1, batch_first=True)
        self.dec_modules = []

        for i in range(3):
            if i==0:
                dec_conv_layer =  nn.Sequential(           
                        nn.Conv1d(dim_pre,
                                dim_pre,
                                kernel_size=5, stride=1,
                                padding=2, dilation=1),
                        nn.BatchNorm1d(dim_pre))
            else:
                dec_conv_layer =  nn.Sequential(           
                        nn.Conv1d(dim_pre,
                                dim_pre,
                                kernel_size=5, stride=1,
                                padding=2, dilation=1),
                        nn.BatchNorm1d(dim_pre))
            self.dec_modules.append(dec_conv_layer)
        self.dec_modules = nn.ModuleList(self.dec_modules)

        self.dec_lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)
        self.dec_linear2 = LinearNorm(1024, 80)
        self.apply(init_weights)
    

    def encode(self, x):
        shape = x.shape
        
        for layer in self.enc_modules:
            x = F.relu(layer(x))
        
        x = x.transpose(1, 2)

        self.enc_lstm.flatten_parameters()

        outputs, _ = self.enc_lstm(x)
        outputs = outputs.reshape(shape[0], -1)

        outputs = F.relu(self.enc_linear(outputs))
        style = self.style(outputs)
        content = self.content(outputs)

        style_mu = style[:,:self.speaker_size]
        style_logvar = style[:,self.speaker_size:]
        content_mu = content[:,:(self.latent_dim-self.speaker_size)]
        content_logvar = content[:,(self.latent_dim-self.speaker_size):]
        
        return style_mu, style_logvar, content_mu, content_logvar

    def _reparameterize(self, mu, logvar, train=True):
        if train:
            epsilon = Variable(torch.empty(logvar.size()).normal_()).cuda()
            std = logvar.mul(0.5).exp_()
            return epsilon.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        
        output = self.dec_pre_linear1(z)
        output = self.dec_pre_linear2(output)
        # print('output dims: ', output.shape)
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
        return output.transpose(-1, -2)

    def forward(self, x1, x2, train=True):
        style_mu1, style_logvar1, content_mu1, content_logvar1 = self.encode(x1)
        z_content1 = self._reparameterize(content_mu1, content_logvar1, train)

        style_mu2, style_logvar2, content_mu2, content_logvar2 = self.encode(x2)
        z_content2 = self._reparameterize(content_mu2, content_logvar2, train)

        style_mu2 = style_mu2.detach()
        style_logvar2 = style_logvar2.detach()
        z_style_mu = (style_mu1 + style_mu2)/2
        z_style_logvar = (style_logvar1 + style_logvar2)/2
        z_style = self._reparameterize(z_style_mu, z_style_logvar)

        z1 = torch.cat((z_style, z_content1), dim=-1)
        z2 = torch.cat((z_style, z_content2), dim=-1)

        ## parameters of distribution of sample 1
        q_z1_mu = torch.cat((z_style_mu, content_mu1), dim=-1)
        q_z1_logvar = torch.cat((z_style_logvar, content_logvar1), dim=-1)

        ## parameters of distribution of sample 2
        q_z2_mu = torch.cat((z_style_mu, content_mu2), dim=-1)
        q_z2_logvar = torch.cat((z_style_logvar, content_logvar2), dim=-1)

        recons_x1 = self.decode(z1)
        recons_x2 = self.decode(z2)

        recons_x1_hat = recons_x1 + self.postnet(recons_x1)
        recons_x2_hat = recons_x2 + self.postnet(recons_x2)        
        return recons_x1, recons_x2, recons_x1_hat, recons_x2_hat,q_z1_mu, q_z1_logvar, q_z2_mu, q_z2_logvar, z_style_mu, z_style_logvar


    def update_c(self):
        self._c += self._c_delta
    
    def update_beta(self):
        self._beta += self._beta_delta

class ConvolutionalMulVAE(VariationalBaseModelVAE):
    def __init__(self, dataset, width, height,
                latent_sz, learning_rate, alpha,
                log_interval, normalize, batch_size, speaker_size,channels=1, device=torch.device("cuda"),
                latent_dim=256, beta=0.1, mse_cof=10, kl_cof=10, style_cof=0.1):
        super().__init__(dataset, width, height, channels,latent_sz, learning_rate,
                        device, log_interval, batch_size)
        self.batch_size = batch_size
        self.alpha = alpha
        self.lr= learning_rate
        self.latent_dim = latent_dim
        self.mse_cof = mse_cof
        self.kl_cof = kl_cof
        self.style_cof = style_cof

        self.model = DisentangledVAE(latent_dim=self.latent_dim, beta=0.1, batch_size=batch_size, speaker_size=speaker_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_losses = []
        self.test_losses = []


################# Original loss function for paper "Weakly-Supervised Disentanglement Without Compromises" #########
    def loss_functionGVAE2(self, x1, x2, x_recon1, x_recon2, recons_x1_hat, recons_x2_hat,
                     q_z1_mu, q_z1_logvar, q_z2_mu, q_z2_logvar, style_mu1, style_logvar1, train=False):  
        
        with torch.autograd.set_detect_anomaly(True):
            MSE_x1 = torch.nn.functional.l1_loss(x1, x_recon1, reduction='sum').div(self.batch_size)
            MSE_x2 = torch.nn.functional.l1_loss(x2, x_recon2, reduction='sum').div(self.batch_size)

            MSE_x1_hat = torch.nn.functional.l1_loss(x1, recons_x1_hat, reduction='sum').div(self.batch_size)
            MSE_x2_hat = torch.nn.functional.l1_loss(x2, recons_x2_hat, reduction='sum').div(self.batch_size)

            z1_kl_loss = (-0.5)*torch.sum(1 + q_z1_logvar - q_z1_mu.pow(2) - q_z1_logvar.exp(), axis=-1).mean()
            z2_kl_loss = (-0.5)*torch.sum(1 + q_z2_logvar - q_z2_mu.pow(2) - q_z2_logvar.exp(), axis=-1).mean()

            z_kl_style = (-1)*torch.sum(1 + style_logvar1 - style_mu1.pow(2) - style_logvar1.exp()).div(self.batch_size)

            LOSS = self.mse_cof*(MSE_x1 + MSE_x2 + MSE_x1_hat + MSE_x2_hat) + self.kl_cof*(z1_kl_loss + z2_kl_loss)
        
        return LOSS, MSE_x1, MSE_x2, MSE_x1_hat, MSE_x2_hat, z1_kl_loss, z2_kl_loss, z_kl_style

    def update_(self):
        self.model.update_c()
        self.model.update_beta()

    ## the KL divergence from paper "PREVENTING POSTERIOR COLLAPSE WITH δ-VAES" #############3
    def compute_KL_delta_VAE(self, mu, logvar, alpha=0.95):
        alpha = Variable(torch.tensor(alpha), requires_grad=False).float().cuda()
        kl_divergence=0
        # for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            if j==0:
                kl_divergence = kl_divergence + (f_function(logvar[:,j].exp()) + mu[:,j].pow(2))
            else:
                kl_divergence = kl_divergence + f_function(logvar[:,j].exp()/(1 - torch.pow(alpha,2)))
                kl_divergence = kl_divergence + ( torch.pow((mu[:,j] - alpha*mu[:,j-1]), 2) + torch.pow(alpha,2)*logvar[:,j-1] )/(1 - torch.pow(alpha,2) )

        return (-0.5)*torch.sum(kl_divergence)
    def update_kl(self):
        self.kl_cof = min(self.kl_cof*2, 10)
    
    def set_kl(self, beta):
        self.kl = beta


def f_function(x,coef=1):
    return coef*x - torch.log(x) - 1
