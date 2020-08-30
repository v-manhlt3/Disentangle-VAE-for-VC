import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F 
from sparse_encoding.variational_base_mulvae_model import VariationalBaseModel
import timeit
from sparse_encoding import utils
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


def accumulate_evidence(mu, logvar, is_cuda, batch_labels):
    var_dict = {}
    mu_dict = {}
    batch_labels = batch_labels.cpu().detach().numpy()
    batch_labels = np.unique(batch_labels)
    # convert logvar to variance for calculations
    var = logvar.exp_()

    # calculation var inverse for each group using group vars
    for i in range(len(batch_labels)):
        group_label = batch_labels[i]

        # remove 0 value from variances 
        for j in range(len(logvar[0])):
            if var[i][j] ==0:
                var[i][j] = 1e-6
        if group_label in var_dict.keys():
            var_dict[group_label]+= (1/var[i])
        else:
            var_dict[group_label] = (1/var[i])

    # invert var inverses to calculate mu and return value
    for group_label in var_dict.keys():
        var_dict[group_label] = 1 / var_dict[group_label]
    
    # calculate mu for each group
    for i in range(len(batch_labels)):
        group_label = batch_labels[i]

        if group_label in mu_dict.keys():
            mu_dict[group_label] += mu[i]*(1/logvar[i])
        else:
            mu_dict[group_label] = mu[i]*(1/logvar[i])
    # multply group var with sums calculated above to get mu for the group
    for group_label in mu_dict.keys():
        mu_dict[group_label] *= var_dict[group_label]
        
    group_mu = torch.FloatTensor(len(mu_dict), mu.shape[1])
    group_var =  torch.FloatTensor(len(var_dict), var.shape[1])

    if is_cuda:
        group_mu.cuda()
        group_var.cuda()

    idx = 0
    for key in var_dict.keys():
        group_mu[idx] = mu_dict[key]
        group_var[idx] = var_dict[key]

        for j in range(len(group_var[idx])):
            if group_var[idx][j] == 0:
                group_var[idx][j] = 1e-6

        idx=idx+1

    return Variable(group_mu, requires_grad=True).cuda(), Variable(torch.log(group_var), requires_grad=True).cuda()
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


class MulVAE(nn.Module):

    def __init__(self, input_sz = (1, 64, 80),
                kernel_szs = [512,512,512],
                hidden_sz: int = 256,
                latent_sz: int = 32,
                c: float = 512,
                c_delta: float = 0.001,
                beta: float = 0.1,
                beta_delta: float = 0,
                dim_neck=64, latent_dim=256, dim_pre=512, batch_size=10):
        super(MulVAE, self).__init__()

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
        # self.postnet = Postnet()

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

        self.enc_linear = LinearNorm(8192, 1024)

        self.style_mu = LinearNorm(1024, 4) # length=64, 8192
        self.style_logvar = LinearNorm(1024, 4)

        self.content_mu = LinearNorm(1024, latent_dim - 4)
        self.content_logvar = LinearNorm(1024, latent_dim - 4)
        ############################ Decoder Architecture ####################
        self.dec_pre_linear1 = nn.Linear(latent_dim, 1024)
        self.dec_pre_linear2 = nn.Linear(1024, 8192)
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
        outputs = outputs.reshape(shape[0], -1)

        outputs = self.enc_linear(outputs)

        style_mu = self.style_mu(outputs)
        # style_logvar = F.tanh(self.style_logvar(outputs))*5
        style_logvar = self.style_logvar(outputs)

        content_mu = self.content_mu(outputs)
        # content_logvar = F.tanh(self.content_logvar(outputs))*5
        content_logvar = self.content_logvar(outputs)
        
        return style_mu, style_logvar, content_mu, content_logvar

    def _reparameterize(self, mu, logvar):
        epsilon = Variable(torch.empty(logvar.size()).normal_()).cuda()
        std = logvar.mul(0.5).exp_()
        return epsilon.mul(std).add_(mu)

    def decode(self, z_content, z_style):
        
        z = torch.cat((z_content, z_style), dim= -1)
        # print('latent size: ',z.shape)
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
        output = F.relu(self.dec_linear2(output))
        return output.transpose(-1, -2)

    def forward(self, x, speaker_ids, train=True):
        style_mu, style_logvar, content_mu, content_logvar  = self.encode(x)
        group_style_mu, group_style_logvar = utils.accumulate_group_evidence(style_mu, style_logvar, speaker_ids, True)
        # print('group style mu shape: ', group_style_mu.shape)
        if train:
            z_content =  self._reparameterize(content_mu, content_logvar)
            # z_style = utils.group_wise_reparameterize(training=True,mu=group_style_mu, logvar=group_style_logvar, labels_batch=speaker_ids, cuda=True)
            z_style = group_style_mu
        else:
            z_content = content_mu
            z_style = group_style_mu
            # z_style = tile(z_style, 0, self.num_utt)
        # print('z_content: ', z_content.shape)
        # print('z_style: ', z_style.shape)
        x_hat0 = self.decode(z_content, z_style)
        # x_hat = x_hat0 + self.postnet(x_hat0)
        return x_hat0, content_mu, content_logvar, group_style_mu, group_style_logvar

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


class ConvolutionalMulVAE(VariationalBaseModel):
    def __init__(self, dataset, width, height,
                latent_sz, learning_rate, alpha,
                log_interval, normalize, batch_size, channels=1, device=torch.device("cuda"),
                latent_dim=256, beta=0.1):
        super().__init__(dataset, width, height, channels,latent_sz, learning_rate,
                        device, log_interval, batch_size)
        self.batch_size = batch_size
        self.alpha = alpha
        self.lr= learning_rate
        self.latent_dim = latent_dim
        # self.hidden_sz = hidden_sz
        # self.kernel_szs = [int(ks) for ks in str(kernel_szs).split(',')]

        self.model = MulVAE(latent_dim=self.latent_dim, beta=0.1, batch_size=batch_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_losses = []
        self.test_losses = []

    # Reconstruction + KL divergence loss sumed over all elements of batch
    def loss_function(self, x, x_recon,
                     content_mu, content_logvar, group_style_mu, group_style_logvar, train=False):
        
        shape = x.shape
        # print('-------------------shape of mel: ', x.shape)
        # print('-------------------shape of recons mel: ', x_recon0.shape)
        # MSE0 = (shape[-1]*shape[-2])*torch.nn.functional.l1_loss(x, x_recon0)
        # MSE = (shape[-1]*shape[-2])*torch.nn.functional.l1_loss(x, x_recon)
        # MSE0 = torch.nn.functional.l1_loss(x, x_recon0, reduction='sum').div(self.batch_size)
        MSE = torch.nn.functional.l1_loss(x, x_recon, reduction='sum').div(self.batch_size)

        # group_style_kl_loss = (-0.5)*torch.sum(1 + group_style_logvar - group_style_mu.pow(2) - group_style_logvar.exp()).div(self.batch_size)
        group_style_kl_loss =  self.compute_KL_delta_VAE(group_style_mu, group_style_logvar).div(self.batch_size)
        content_kl_loss = (-0.5)*torch.sum(1 + content_logvar - content_mu.pow(2) - content_logvar.exp()).div(self.batch_size)

        LOSS =  MSE + self.model._beta*group_style_kl_loss + self.model._beta*content_kl_loss
        
        return LOSS,MSE, group_style_kl_loss, content_kl_loss

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


def f_function(x,coef=1):
    return coef*x - torch.log(x) - 1

# model = ConvVSC()
# model = model.cuda()
# data = torch.randn(10, 64, 80).cuda()

# output = model(data)
# print(output[0].shape)




