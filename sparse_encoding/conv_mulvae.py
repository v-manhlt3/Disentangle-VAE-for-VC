import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F 
# from sparse_encoding.variational_base_mulvae_model import VariationalBaseModel
from sparse_encoding.variational_base_acvae import VariationalBaseModelGVAE
# from variational_base_acvae import VariationalBaseModelGVAE
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
                dim_neck=64, latent_dim=64, dim_pre=512, batch_size=10):
        super(MulVAE, self).__init__()

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
        # self.mse_cof = mse_cof
        # self.kl_cof = kl_cof
        # self.style_cof = style_cof
        # self.postnet = Postnet()

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

        self.enc_linear = LinearNorm(2048, 256)

        # self.style_mu = LinearNorm(256, 3) # length=64, 8192
        # self.style_logvar = LinearNorm(256, 3)

        # self.content_mu = LinearNorm(256, latent_dim - 3)
        # self.content_logvar = LinearNorm(256, latent_dim - 3)
        self.mu = LinearNorm(256, latent_dim)
        self.logvar = LinearNorm(256, latent_dim)
        ############################ Decoder Architecture ####################
        self.dec_pre_linear1 = nn.Linear(latent_dim, 256)
        self.dec_pre_linear2 = nn.Linear(256, 2048)
        self.dec_lstm1 = nn.LSTM(dim_neck*2, 512, 1, batch_first=True)
        self.dec_modules = []

        for i in range(3):
            if i==0:
                dec_conv_layer =  nn.Sequential(           
                        nn.ConvTranspose1d(dim_pre,
                                dim_pre,
                                kernel_size=5, stride=2,
                                padding=1, dilation=1),
                        nn.BatchNorm1d(dim_pre))
            else:
                dec_conv_layer =  nn.Sequential(           
                        nn.ConvTranspose1d(dim_pre,
                                dim_pre,
                                kernel_size=5, stride=2,
                                padding=2, dilation=1),
                        nn.BatchNorm1d(dim_pre))
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

        outputs = F.relu(self.enc_linear(outputs))
        mu = self.mu(outputs)
        logvar = self.logvar(outputs)

        style_mu = mu[:,:4]
        style_logvar = logvar[:,:4]

        content_mu = mu[:,4:]
        content_logvar = logvar[:,4:] 
        
        return style_mu, style_logvar, content_mu, content_logvar

    def _reparameterize(self, mu, logvar, train=True):
        if train:
            epsilon = Variable(torch.empty(logvar.size()).normal_()).cuda()
            std = logvar.mul(0.5).exp_()
            return epsilon.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        
        # z = torch.cat((z_content, z_style), dim= -1)
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
        output = self.dec_linear2(output)
        return output.transpose(-1, -2)[:,:,:-1]

################## for MulVAE #######################################################
    # def forward(self, x, speaker_ids, train=True):
    #     style_mu, style_logvar, content_mu, content_logvar  = self.encode(x)
    #     group_style_mu, group_style_logvar = utils.accumulate_group_evidence(style_mu, style_logvar, speaker_ids, True)
    #     # print('group style mu shape: ', group_style_mu.shape)
    #     if train:
    #         z_content =  self._reparameterize(content_mu, content_logvar)
    #         # z_style = utils.group_wise_reparameterize(training=True,mu=group_style_mu, logvar=group_style_logvar, labels_batch=speaker_ids, cuda=True)
    #         z_style = group_style_mu
    #     else:
    #         z_content = content_mu
    #         z_style = group_style_mu
    #         # z_style = tile(z_style, 0, self.num_utt)
    #     # print('z_content: ', z_content.shape)
    #     # print('z_style: ', z_style.shape)
    #     x_hat0 = self.decode(z_content, z_style)
    #     # x_hat = x_hat0 + self.postnet(x_hat0)
    #     return x_hat0, content_mu, content_logvar, group_style_mu, group_style_logvar

##################### For Group VAE ##################################################
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
        # return recons_x1, recons_x2, content_mu1, content_logvar1, content_mu2, content_logvar2, style_mu1, style_logvar1
        return recons_x1, recons_x2, q_z1_mu, q_z2_logvar, q_z2_mu, q_z2_logvar, style_mu1, style_logvar1
#######################################################################################

    def update_c(self):
        self._c += self._c_delta
    
    def update_beta(self):
        self._beta += self._beta_delta

class ConvolutionalMulVAE(VariationalBaseModelGVAE):
    def __init__(self, dataset, width, height,
                latent_sz, learning_rate, alpha,
                log_interval, normalize, batch_size, channels=1, device=torch.device("cuda"),
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
        # self.hidden_sz = hidden_sz
        # self.kernel_szs = [int(ks) for ks in str(kernel_szs).split(',')]

        self.model = MulVAE(latent_dim=self.latent_dim, beta=0.1, batch_size=batch_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_losses = []
        self.test_losses = []

    # Reconstruction + KL divergence loss sumed over all elements of batch
    def loss_functionMulVAE(self, x, x_recon,
                     content_mu, content_logvar, group_style_mu, group_style_logvar, train=False):
        
        # shape = x.shape
        MSE = torch.nn.functional.l1_loss(x, x_recon, reduction='sum').div(self.batch_size)

        group_style_kl_loss = (-0.5)*torch.sum(1 + group_style_logvar - group_style_mu.pow(2) - group_style_logvar.exp()).div(self.batch_size)
        # group_style_kl_loss =  self.compute_KL_delta_VAE(group_style_mu, group_style_logvar).div(self.batch_size)
        content_kl_loss = (-0.5)*torch.sum(1 + content_logvar - content_mu.pow(2) - content_logvar.exp()).div(self.batch_size)

        LOSS =  MSE + self.model._beta*group_style_kl_loss + self.model._beta*content_kl_loss
        
        return LOSS,MSE, group_style_kl_loss, content_kl_loss
    
########################################## loss function for GVAE ##################################
    def loss_functionGVAE(self, x1, x2, x_recon1, x_recon2,
                     q_z1_mu, q_z1_logvar, q_z2_mu, q_z2_logvar, style_mu1, style_logvar1,train=False):  
        
        with torch.autograd.set_detect_anomaly(True):
            # MSE0 = torch.nn.functional.l1_loss(x, x_recon0, reduction='sum').div(self.batch_size)
            MSE_x1 = torch.nn.functional.l1_loss(x1, x_recon1, reduction='sum').div(self.batch_size)
            MSE_x2 = torch.nn.functional.l1_loss(x2, x_recon2, reduction='sum').div(self.batch_size)

            z1_kl_loss = (-0.5)*torch.sum(1 + q_z1_logvar - q_z1_mu.pow(2) - q_z1_logvar.exp()).div(self.batch_size)
            z2_kl_loss = (-0.5)*torch.sum(1 + q_z2_logvar - q_z2_mu.pow(2) - q_z2_logvar.exp()).div(self.batch_size)

            z_kl_style = (-1)*torch.sum(1 + style_logvar1 - style_mu1.pow(2) - style_logvar1.exp()).div(self.batch_size)
            # z_kl_style = torch.sum(style_mu1).div(self.batch_size)

            # LOSS = 10*MSE_x1 + 10*MSE_x2 + 10*z1_kl_loss + 10*z2_kl_loss + 0.1*z_kl_style
            LOSS = self.mse_cof*(MSE_x1 + MSE_x2) + self.kl_cof*(z1_kl_loss + z2_kl_loss) + self.style_cof*z_kl_style
        
        return LOSS, MSE_x1, MSE_x2, z1_kl_loss, z2_kl_loss, z_kl_style

################# Original loss function for paper "Weakly-Supervised Disentanglement Without Compromises" #########
    def loss_functionGVAE2(self, x1, x2, x_recon1, x_recon2,
                     q_z1_mu, q_z1_logvar, q_z2_mu, q_z2_logvar, style_mu1, style_logvar1, train=False):  
        
        with torch.autograd.set_detect_anomaly(True):
            # MSE0 = torch.nn.functional.l1_loss(x, x_recon0, reduction='sum').div(self.batch_size)
            # MSE_x1 = torch.nn.functional.l1_loss(x1, x_recon1, reduction='sum').div(self.batch_size)
            # MSE_x2 = torch.nn.functional.l1_loss(x2, x_recon2, reduction='sum').div(self.batch_size)

            MSE_x1 = torch.nn.functional.l1_loss(x1, x_recon1, reduction='mean')
            MSE_x2 = torch.nn.functional.l1_loss(x2, x_recon2, reduction='mean')

            z1_kl_loss = (-0.5)*torch.sum(1 + q_z1_logvar - q_z1_mu.pow(2) - q_z1_logvar.exp()).div(self.batch_size)
            z2_kl_loss = (-0.5)*torch.sum(1 + q_z2_logvar - q_z2_mu.pow(2) - q_z2_logvar.exp()).div(self.batch_size)

            z_kl_style = (-1)*torch.sum(1 + style_logvar1 - style_mu1.pow(2) - style_logvar1.exp()).div(self.batch_size)
            # z_kl_style = torch.sum(style_mu1).div(self.batch_size)

            # LOSS = 10*MSE_x1 + 10*MSE_x2 + 10*z1_kl_loss + 10*z2_kl_loss + 0.1*z_kl_style
            LOSS = self.mse_cof*(MSE_x1 + MSE_x2) + self.kl_cof*(z1_kl_loss + z2_kl_loss)
        
        return LOSS, MSE_x1, MSE_x2, z1_kl_loss, z2_kl_loss, z_kl_style

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

# data = torch.randn(10, 80, 64).cuda()
# data2 = torch.randn(10, 80, 64).cuda()
# model = MulVAE().cuda()
# output = model(data,data2)

# print('output shape: ', output[0].shape)
# model = ConvVSC()
# model = model.cuda()
# data = torch.randn(10, 64, 80).cuda()

# output = model(data)
# print(output[0].shape)




