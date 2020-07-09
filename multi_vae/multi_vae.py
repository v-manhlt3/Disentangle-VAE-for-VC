import torch
import torch.nn as nn
from collections import OrderedDict

from itertools import cycle
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
import torchvision
import matplotlib.pyplot as plt
import numpy as np

transform_config = Compose([ToTensor()])

class Encoder(nn.Module):
    def __init__(self, style_dim, class_dim):
        super(Encoder, self).__init__()

        self.linear_model = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(in_features=784, out_features=500, bias=True)),
            ('tanh_1', nn.Tanh())
        ]))

        #style
        self.style_mu = nn.Linear(in_features=500, out_features=style_dim, bias=True)
        self.style_logvar = nn.Linear(in_features=500, out_features=style_dim, bias=True)

        self.class_mu = nn.Linear(in_features=500, out_features=class_dim, bias=True)
        self.class_logvar = nn.Linear(in_features=500, out_features=class_dim, bias=True)

    
    def forward(self, x):
        x = x.view(x.size(0), x.size(1)*x.size(2)*x.size(3))
        x = self.linear_model(x)

        style_latent_space_mu = self.style_mu(x)
        style_latent_space_logvar = self.style_logvar(x)

        class_latent_mu = self.class_mu(x)
        class_latent_logvar = self.class_logvar(x)

        return style_latent_space_mu, style_latent_space_logvar, class_latent_mu, class_latent_logvar

class Decoder(nn.Module):
    def __init__(self, style_dim, class_dim):
        super(Decoder, self).__init__()

        self.linear_model = nn.Sequential(OrderedDict([
            ('linear_1', nn.Linear(in_features=style_dim + class_dim, out_features=500, bias=True)),
            ('tanh_1', nn.Tanh()),

            ('linear_2', nn.Linear(in_features=500, out_features=784, bias=True)),
            ('sigmoid_final', nn.Sigmoid())
        ]))

    def _reparameterize(self, mu, logvar, training):
        if training:
            epsilon = Variable(torch.empty(logvar.size()).normal_())
            std = logvar.mul(0.5).exp_()
            return epsilon.mul(std).add_(mu)
        else:
            return mu
    
    def forward(self, style_latent_space, class_latent_space):
        x = torch.cat((style_latent_space, class_latent_space), dim=1)

        x = self.linear_model(x)
        x = x.view(x.size(0), 1, 28, 28)
        return x

class Classifier(nn.Module):
    def __init__(self, z_dim, num_classes):
        super(Classifier, self).__init__()

        self.fc_model = nn.Sequential(
            nn.Linear(z_dim, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 256, bias=True),
            nn.BatchNorm1d(num_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, num_classes, bias=True)
        )

    def forward(self, z):
        # z = self._reparameterize(mu, logvar)

        x = self.fc_model(z)
        return x

if __name__ =="__main__":

    encoder = Encoder(10, 10)
    decoder = Decoder(10, 10)
    classifier = Classifier(z_dim=16, num_classes=10)

    mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
    loader = cycle(DataLoader(mnist, batch_size=64, shuffle=True, num_workers=0, drop_last=True))

    image_batch, label_batch = next(loader)

    style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(image_batch))

    style_space = decoder._reparameterize(style_mu, style_logvar, True)
    class_space = decoder._reparameterize(class_mu, class_logvar, True)

    reconstructed_image = decoder(style_space, class_space).detach()
    # reconstructed_image = reconstructed_image.permute(0,2,3,1)
    # reconstructed_image = np.squeeze(reconstructed_image, 1)
    print(reconstructed_image.shape)
    grid = torchvision.utils.make_grid(reconstructed_image, nrow=8)
    # grid = grid.permute(0,2,3,1)
    print(grid.shape)
    plt.imshow(grid.permute(1,2,0))
    plt.show()