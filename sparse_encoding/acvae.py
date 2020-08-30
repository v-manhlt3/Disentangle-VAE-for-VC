import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import timeit
from sparse_encoding import utils
# import utils
from torch.autograd import Variable
from sparse_encoding.variational_base_mulvae_model import VariationalBaseModel


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
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1),
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal



class ACVAE(nn.Module):
    def __init__(self):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super(ACVAE, self).__init__()
        self._beta = 0.1
        # self.postnet = Postnet()
        # Encoder
        self.conv1 = nn.Conv2d(1, 8, (3,9), (1,1), padding=(1, 4))
        self.conv1_bn = nn.BatchNorm2d(8)
        self.conv1_gated = nn.Conv2d(1, 8, (3,9), (1,1), padding=(1, 4))
        self.conv1_gated_bn = nn.BatchNorm2d(8)
        self.conv1_sigmoid = nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(8, 16, (4,8), (2,2), padding=(1, 3))
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv2_gated = nn.Conv2d(8, 16, (4,8), (2,2), padding=(1, 3))
        self.conv2_gated_bn = nn.BatchNorm2d(16)
        self.conv2_sigmoid = nn.Sigmoid()
        
        self.conv3 = nn.Conv2d(16, 16, (4,8), (2,2), padding=(1, 3))
        self.conv3_bn = nn.BatchNorm2d(16)
        self.conv3_gated = nn.Conv2d(16, 16, (4,8), (2,2), padding=(1, 3))
        self.conv3_gated_bn = nn.BatchNorm2d(16)
        self.conv3_sigmoid = nn.Sigmoid()
        
        self.conv4_style = nn.Conv2d(16, 10//2, (9,5), (9,1), padding=(1, 2))
        # self.conv4_style_logvar = nn.Conv2d(16, 10//2, (9,5), (9,1), padding=(1, 2))

        self.conv4_content = nn.Conv2d(16, 10//2, (9,5), (9,1), padding=(1, 2))
        # self.conv4_content_logvar = nn.Conv2d(16, 10//2, (9,5), (9,1), padding=(1, 2))
        
        # Decoder
        self.upconv1 = nn.ConvTranspose2d(5, 16, (9,5), (9,1), padding=(0, 2))
        self.upconv1_bn = nn.BatchNorm2d(16)
        self.upconv1_gated = nn.ConvTranspose2d(5, 16, (9,5), (9,1), padding=(0, 2))
        self.upconv1_gated_bn = nn.BatchNorm2d(16)
        self.upconv1_sigmoid = nn.Sigmoid()
        
        self.upconv2 = nn.ConvTranspose2d(16, 16, (4,8), (2,2), padding=(0, 3))
        self.upconv2_bn = nn.BatchNorm2d(16)
        self.upconv2_gated = nn.ConvTranspose2d(16, 16, (4,8), (2,2), padding=(0, 3))
        self.upconv2_gated_bn = nn.BatchNorm2d(16)
        self.upconv2_sigmoid = nn.Sigmoid()
        
        self.upconv3 = nn.ConvTranspose2d(16, 8, (4,8), (2,2), padding=(0, 3))
        self.upconv3_bn = nn.BatchNorm2d(8)
        self.upconv3_gated = nn.ConvTranspose2d(16, 8, (4,8), (2,2), padding=(0, 3))
        self.upconv3_gated_bn = nn.BatchNorm2d(8)
        self.upconv3_sigmoid = nn.Sigmoid()
        
        self.upconv4_mu = nn.ConvTranspose2d(8, 2//2, (3,9), (1,1), padding=(0, 4))
        

    def encode(self, x):
        x = x.unsqueeze(1)

        h1_ = self.conv1_bn(self.conv1(x))
        h1_gated = self.conv1_gated_bn(self.conv1_gated(x))
        h1 = torch.mul(h1_, self.conv1_sigmoid(h1_gated)) 
        
        h2_ = self.conv2_bn(self.conv2(h1))
        h2_gated = self.conv2_gated_bn(self.conv2_gated(h1))
        h2 = torch.mul(h2_, self.conv2_sigmoid(h2_gated)) 
        
        h3_ = self.conv3_bn(self.conv3(h2))
        h3_gated = self.conv3_gated_bn(self.conv3_gated(h2))
        h3 = torch.mul(h3_, self.conv3_sigmoid(h3_gated)) 
        
        style = self.conv4_style(h3)
        content = self.conv4_content(h3)

        shape = style.shape
        style_flatten = style.view(shape[0], shape[1]*shape[2]*shape[3])
        content_flatten = content.view(shape[0], shape[1]*shape[2]*shape[3])
        # print('h4 mu shape: ', h4_mu.shape)
        
        latent_dim = style_flatten.shape[1]  
        style_mu = style_flatten[:, :latent_dim//2]
        style_logvar = style_flatten[:, :latent_dim//2]

        content_mu = content_flatten[:, latent_dim//2:]
        content_logvar = content_flatten[:, latent_dim//2:]
       
        return style_mu, style_logvar, content_mu, content_logvar 

    def decode(self, z_style, z_content):
        
        z = torch.cat((z_content, z_style), dim= -1)
        z = z.view(z.shape[0],5,2,16)
        h5_ = self.upconv1_bn(self.upconv1(z))
        h5_gated = self.upconv1_gated_bn(self.upconv1(z))
        h5 = torch.mul(h5_, self.upconv1_sigmoid(h5_gated)) 
        
        h6_ = self.upconv2_bn(self.upconv2(h5))
        h6_gated = self.upconv2_gated_bn(self.upconv2(h5))
        h6 = torch.mul(h6_, self.upconv2_sigmoid(h6_gated)) 
        
        h7_ = self.upconv3_bn(self.upconv3(h6))
        h7_gated = self.upconv3_gated_bn(self.upconv3(h6))
        h7 = torch.mul(h7_, self.upconv3_sigmoid(h7_gated)) 
        
        h8_mu = self.upconv4_mu(h7)
        h8_mu = h8_mu.squeeze(1)
        # h8_logvar = self.upconv4_logvar(h7)
        
        return h8_mu  
        
    
    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, speaker_ids, train=True):
        style_mu, style_logvar, content_mu, content_logvar = self.encode(x)
        # print('style mu: ', style_mu.shape)
        group_style_mu, group_style_logvar = utils.accumulate_group_evidence(style_mu, style_logvar, speaker_ids, True)

        if train:
            z_content =  self._reparameterize(content_mu, content_logvar)
            z_style = utils.group_wise_reparameterize(training=True,mu=group_style_mu, logvar=group_style_logvar, labels_batch=speaker_ids, cuda=True)
            # z_style = group_style_mu
        else:
            z_content = content_mu
            z_style = group_style_mu

        recons_x0 = self.decode(z_style, z_content)
        # recons_x = recons_x0 + self.postnet(recons_x0)
        return recons_x0, content_mu, content_logvar, group_style_mu, group_style_logvar
    
    



class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(1, 16,
                         kernel_size=(5,9), stride=(1,1),
                         padding=(2,4),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm2d(16))
        )

        for i in range(1, 5 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(16,
                             16,
                             kernel_size=(5,9), stride=(1,1),
                             padding=(2,4),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm2d(16))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(16, 1,
                         kernel_size=(5,9), stride=(1,1),
                         padding=(2,4),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm2d(1))
            )

    def forward(self, x):
        x = x.unsqueeze(1)
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x

class ConvolutionalACVAE(VariationalBaseModel):
    def __init__(self, dataset, width, height,
                latent_sz, learning_rate, alpha,
                log_interval, normalize, batch_size,channels=1, device=torch.device("cuda"),
                latent_dim=256, beta=0.1):
        super().__init__(dataset, width, height, channels,latent_sz, learning_rate,
                        device, log_interval, batch_size)
        
        self.batch_size = batch_size
        self.alpha = alpha
        self.lr= learning_rate
        self.latent_dim = latent_dim
        # self.gamma = gamma
        # self.hidden_sz = hidden_sz
        # self.kernel_szs = [int(ks) for ks in str(kernel_szs).split(',')]

        self.model = ACVAE().to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)



    # Reconstruction + KL divergence loss sumed over all elements of batch
    def loss_function(self, x, x_recon,
                     content_mu, content_logvar, group_style_mu, group_style_logvar, train=False):
        
        with torch.autograd.set_detect_anomaly(True):
            # MSE0 = torch.nn.functional.l1_loss(x, x_recon0, reduction='sum').div(self.batch_size)
            MSE = torch.nn.functional.l1_loss(x, x_recon, reduction='sum').div(self.batch_size)

            group_style_kl_loss = (-0.5)*torch.sum(1 + group_style_logvar - group_style_mu.pow(2) - group_style_logvar.exp()).div(self.batch_size)
            # group_style_kl_loss =  self.compute_KL_delta_VAE(group_style_mu, group_style_logvar).div(self.batch_size)
            # content_kl_loss = 0
            content_kl_loss = (-0.5)*torch.sum(1 + content_logvar - content_mu.pow(2) - content_logvar.exp()).div(self.batch_size)

            LOSS = MSE + self.model._beta*group_style_kl_loss + self.model._beta*content_kl_loss
        
        return LOSS, MSE, group_style_kl_loss, content_kl_loss

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

if __name__=='__main__':

    data = torch.randn(10,80,64).cuda()
    spk_ids = torch.randn(10,1).cuda()
    model = ACVAE().cuda()

    output = model(data, spk_ids)
    print(output[0].shape)
    print(output[1].shape)