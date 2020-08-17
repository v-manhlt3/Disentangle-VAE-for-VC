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
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

# class Postnet(nn.Module):
#     """Postnet
#         - Five 1-d convolution with 512 channels and kernel size 5
#     """

#     def __init__(self):
#         super(Postnet, self).__init__()
#         self.convolutions = nn.ModuleList()

#         self.convolutions.append(
#             nn.Sequential(
#                 ConvNorm(80, 512,
#                          kernel_size=5, stride=1,
#                          padding=2,
#                          dilation=1, w_init_gain='tanh'),
#                 nn.BatchNorm1d(512))
#         )

#         for i in range(1, 5 - 1):
#             self.convolutions.append(
#                 nn.Sequential(
#                     ConvNorm(512,
#                              512,
#                              kernel_size=5, stride=1,
#                              padding=2,
#                              dilation=1, w_init_gain='tanh'),
#                     nn.BatchNorm1d(512))
#             )

#         self.convolutions.append(
#             nn.Sequential(
#                 ConvNorm(512, 80,
#                          kernel_size=5, stride=1,
#                          padding=2,
#                          dilation=1, w_init_gain='linear'),
#                 nn.BatchNorm1d(80))
#             )

#     def forward(self, x):
#         for i in range(len(self.convolutions) - 1):
#             x = torch.tanh(self.convolutions[i](x))

#         x = self.convolutions[-1](x)

#         return x 

class ACVAE(nn.Module):
    def __init__(self, input_sz = (1, 64, 80),
                kernel_szs = [512,512,512],
                hidden_sz: int = 256,
                latent_sz: int = 32,
                c: float = 50,
                c_delta: float = 0.001,
                beta: float = 0.1,
                beta_delta: float = 0,
                dim_neck=64, latent_dim=256, dim_pre=512):
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
        # self.label_num = nb_label
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super(ACVAE, self).__init__()
        
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
        
        self.conv4_mu = nn.Conv2d(16, 10//2, (9,5), (9,1), padding=(1, 2))
        self.conv4_logvar = nn.Conv2d(16, 10//2, (9,5), (9,1), padding=(1, 2))
        self.conv4_logspike = nn.Conv2d(16, 10//2, (9,5), (9,1), padding=(1, 2))
        
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
        
        self.output_layer = nn.ConvTranspose2d(8, 2//2, (3,9), (1,1), padding=(0, 4))
        self.apply(init_weights)
        # self.upconv4_logvar = nn.ConvTranspose2d(8, 2//2, (3,9), (1,1), padding=(1, 4))

    def encode(self, x):
       
        h1_ = self.conv1_bn(self.conv1(x))
        h1_gated = self.conv1_gated_bn(self.conv1_gated(x))
        h1 = torch.mul(h1_, self.conv1_sigmoid(h1_gated)) 
        
        h2_ = self.conv2_bn(self.conv2(h1))
        h2_gated = self.conv2_gated_bn(self.conv2_gated(h1))
        h2 = torch.mul(h2_, self.conv2_sigmoid(h2_gated)) 
        
        h3_ = self.conv3_bn(self.conv3(h2))
        h3_gated = self.conv3_gated_bn(self.conv3_gated(h2))
        h3 = torch.mul(h3_, self.conv3_sigmoid(h3_gated)) 
        
        mu = self.conv4_mu(h3)
        logvar = self.conv4_logvar(h3)
        logspike = -F.relu(-self.conv4_logspike(h3))
       
        return mu, logvar, logspike

    def decode(self, z):
        
        h5_ = self.upconv1_bn(self.upconv1(z))
        h5_gated = self.upconv1_gated_bn(self.upconv1(z))
        h5 = torch.mul(h5_, self.upconv1_sigmoid(h5_gated)) 
        
        h6_ = self.upconv2_bn(self.upconv2(h5))
        h6_gated = self.upconv2_gated_bn(self.upconv2(h5))
        h6 = torch.mul(h6_, self.upconv2_sigmoid(h6_gated)) 
        
        h7_ = self.upconv3_bn(self.upconv3(h6))
        h7_gated = self.upconv3_gated_bn(self.upconv3(h6))
        h7 = torch.mul(h7_, self.upconv3_sigmoid(h7_gated)) 
        
        output = F.relu(self.output_layer(h7))
        # h8_logvar = self.upconv4_logvar(h7)
        
        return output
    
        
    
    def reparameterize(self, mu, logvar, logspike):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        gaussian =  eps.mul(std).add_(mu)
        eta = torch.randn_like(std)
        selection = F.sigmoid(self._c*(eta +  logspike.exp()-1))

        return selection.mul(gaussian)

    def forward(self, x):
        x = x.unsqueeze(1)
        mu, logvar, logspike = self.encode(x)
        z = self.reparameterize(mu, logvar, logspike)
        return self.decode(z).squeeze(1), mu, logvar, logspike
    
    def update_c(self):
        self._c += self._c_delta
    
    def update_beta(self):
        self._beta += self._beta_delta

class ConvolutionalACVAE(VariationalBaseModel):
    def __init__(self, dataset, width, height,
                latent_sz, learning_rate, alpha,
                log_interval, normalize, channels=1, device=torch.device("cuda")):
        super().__init__(dataset, width, height, channels,latent_sz, learning_rate,
                        device, log_interval)
        
        self.alpha = alpha
        self.lr= learning_rate
        # self.hidden_sz = hidden_sz
        # self.kernel_szs = [int(ks) for ks in str(kernel_szs).split(',')]

        self.model = ACVAE().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_losses = []
        self.test_losses = []

    # Reconstruction + KL divergence loss sumed over all elements of batch
    def loss_function(self, x, x_recon, mu, logvar, logspike, train=False):
        
        shape = x.shape
        # print('-------------------shape of mel: ', x.shape)
        # print('-------------------shape of recons mel: ', x_recon.shape)
        MSE = (shape[-1]*shape[-2])*torch.nn.functional.l1_loss(x, x_recon)

        spike= torch.clamp(logspike.exp(), 1e-6, 1.0 - 1e-6)

        prior1 = -0.5 * torch.sum(spike.mul(1 + logvar - mu.pow(2) - logvar.exp()))
        prior21 = (1 - spike).mul(torch.log((1 - spike) / (1 - self.alpha)))
        prior22 = spike.mul(torch.log(spike /  self.alpha))
        prior2 = torch.sum(prior21 + prior22)
        PRIOR = prior1 + prior2

        LOSS = MSE + self.model._beta*PRIOR

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
        
        return LOSS, MSE, PRIOR

    def update_(self):
        self.model.update_c()
        self.model.update_beta()
# data = torch.randn(10, 1,80, 64)
# model = ACVAE()
# output = model(data)
# print(output[0].shape)