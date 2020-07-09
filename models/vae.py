import torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)


class Encoder(nn.Module):
    def __init__(self, latent_dim, input_channels):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        model = []
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 160, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(160),
            # nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(160, 160, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(160),
            # nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(160, 160, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(160),
            # nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(160, 160, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(160),
            # nn.ReLU()
        )

        self.style_mu = nn.Linear(160*9, latent_dim//2)
        self.style_logvar = nn.Linear(160*9, latent_dim//2)

        self.content_mu = nn.Linear(160*9, latent_dim//2)
        self.content_logvar = nn.Linear(160*9, latent_dim//2)
        self.apply(init_weights)
        # init weight

    def forward(self, inputs):
        shape = inputs.shape
        # for layer in self.model:
        #     out = layer(inputs)
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        style_mu = self.style_mu(out.view(shape[0], -1))
        style_logvar = self.style_logvar(out.view(shape[0], -1))

        content_mu = self.content_mu(out.view(shape[0], -1))
        content_logvar = self.content_logvar(out.view(shape[0], -1))
        

        return style_mu, style_logvar, content_mu, content_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels, length_output):
        super(Decoder, self).__init__()

        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.length_output = length_output
        self.linear_layer = nn.Linear(latent_dim, 160*9)

        self.trans_conv1 = nn.Sequential(
            nn.ConvTranspose1d(160, 160, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(160),
            # nn.ReLU()
        )
        self.trans_conv2 = nn.Sequential(
            nn.ConvTranspose1d(160, 160, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(160),
            # nn.ReLU()
        )
        self.trans_conv3 = nn.Sequential(
            nn.ConvTranspose1d(160, 160, kernel_size=5, stride=2),
            nn.BatchNorm1d(160),
            # nn.ReLU()
        )
        self.trans_conv4 = nn.Sequential(
            nn.ConvTranspose1d(160, 80, kernel_size=5, stride=1),
            nn.BatchNorm1d(80),
            # nn.ReLU()
        )
        self.linear = nn.Linear(self.length_output, self.length_output)
        self.apply(init_weights)

    def reparameterization(self, mu, logvar, train=True):
        if train:
            eps = torch.autograd.Variable(torch.empty(logvar.shape).normal_())
            std = logvar.mul(0.5).exp_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, style_mu, style_logvar, content_mu, content_logvar):
        style = self.reparameterization(style_mu, style_logvar)
        content = self.reparameterization(content_mu, content_logvar)

        z = torch.cat((style, content), dim=-1)
        shape = z.shape
        out = self.linear_layer(z)
        out = out.view(shape[0], 160, 9)

        out = self.trans_conv1(out)
        out = self.trans_conv2(out)
        out = self.trans_conv3(out)
        out = self.trans_conv4(out)

        out = self.linear(out[:,:,:self.length_output])
        return out[:,:,:self.length_output]

# encoder = Encoder(128, 80)
# decoder = Decoder(128, 80, 81)
# data = torch.rand(5, 80, 81)
# mu1, logvar1, mu2, logvar2 = encoder(data)

# reconstruct = decoder(mu1, logvar1, mu2, logvar2)

# print(reconstruct.shape)
