import torch
import torch.nn as nn

#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    if type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

def speaker_to_onehot(speaker_ids, speaker_all,num_classes=109):
    onehot_embedding = torch.nn.functional.one_hot(torch.arange(0,109), num_classes=109)
    speaker_onehot = []
    for speaker in speaker_ids:
        idx = speaker_all.index(speaker)
        vector = onehot_embedding[idx]
        speaker_onehot.append(vector)

    return torch.tensor(speaker_onehot) 


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

class Encoder(nn.Module):
    def __init__(self, latent_dim, dim_neck,input_channels, freq=16):
        super(Encoder, self).__init__()

        self.latent_dim = latent_dim
        self.freq = freq
        self.dim_neck = dim_neck
        model = []
        self.conv1 = nn.Sequential(
            ConvNorm(input_channels, 512, kernel_size=5, stride=1, padding=2, w_init_gain='relu'),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            ConvNorm(512, 512, kernel_size=5, stride=2, padding=2, w_init_gain='relu'),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            ConvNorm(512, 512, kernel_size=5, stride=2, padding=2, w_init_gain='relu'),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

        # self.style_mu = nn.Linear(16*dim_neck*2, latent_dim//2)
        # self.style_logvar = nn.Linear(16*dim_neck*2, latent_dim//2)

        # self.content_mu = nn.Linear(16*dim_neck*2, latent_dim//2)
        # self.content_logvar = nn.Linear(16*dim_neck*2, latent_dim//2)
        self.style_mu = nn.Linear(1088*2, latent_dim//2)
        self.style_logvar = nn.Linear(1088*2, latent_dim//2)

        self.content_mu = nn.Linear(1088*2, latent_dim//2)
        self.content_logvar = nn.Linear(1088*2, latent_dim//2)

        self.apply(init_weights)
        # init weight

    def forward(self, inputs):
        shape = inputs.shape
        
        # for layer in self.model:
        #     out = layer(inputs)
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.transpose(1,2)

        self.lstm.flatten_parameters()

        shape1 = out.shape
        batch = shape[0]
        sequence_length = shape1[2]
        input_sq_size = shape[1]

        outputs, _ = self.lstm(out)
        outs_forward = outputs[:,:,:self.dim_neck]
        outs_backward = outputs[:,:,self.dim_neck:]

        outputs = torch.cat((outs_backward, outs_forward), dim=-1)
        outputs = outputs.view(shape[0], -1)

        style_mu = self.style_mu(outputs)
        style_logvar = self.style_logvar(outputs)

        content_mu = self.content_mu(outputs)
        content_logvar = self.content_logvar(outputs)
        
        # return codes
        return style_mu, style_logvar, content_mu, content_logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_channels, length_output, dim_neck=64, dim_pre=512):
        super(Decoder, self).__init__()
        self.dim_neck = dim_neck
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.length_output = length_output
        # self.linear_layer = nn.Linear(latent_dim, 33*dim_neck*2)
        self.linear_layer = nn.Linear(latent_dim, 1088*2)

        self.lstm1 = nn.LSTM(dim_neck*2, dim_pre, 1, batch_first=True)
        ###################### Upsampling #######################################
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(512, 512, kernel_size=5, stride=2),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(512, 512, kernel_size=5, stride=2),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        #########################################################################
        self.lstm2 = nn.LSTM(dim_pre, 1024, 2, batch_first=True)

        self.linear_projection = LinearNorm(1024, 80)

        self.apply(init_weights)

    def reparameterization(self, mu, logvar, train=True):
        if train:
            mu=mu.to(device)
            logvar=logvar.to(device)

            eps = torch.autograd.Variable(torch.empty(logvar.shape).normal_(), requires_grad=False)
            eps = torch.tensor(eps, device=device)

            std = logvar.mul(0.5).exp_().to(device)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, style_mu, style_logvar, content_mu, content_logvar, inference=False):
        if inference:
            style = style_mu
            content = content_mu
        else:
            style = self.reparameterization(style_mu, style_logvar)
            content = self.reparameterization(content_mu, content_logvar)

        z = torch.cat((style, content), dim=-1)
        shape = z.shape
        out = self.linear_layer(z)
        out = out.view(shape[0],-1, self.dim_neck*2)


        out_lstm1,_ = self.lstm1(out)
        out_lstm1 = out_lstm1.transpose(1,2)


        out_conv1 = self.conv1(out_lstm1)
        out_conv2 = self.conv2(out_conv1)
        # out_conv3 = self.conv3(out_conv2)

        out_conv2 = out_conv2.transpose(1, 2)
        outputs, _ = self.lstm2(out_conv2)

        decoder_output = torch.nn.functional.sigmoid(self.linear_projection(outputs))
        decoder_output = decoder_output.transpose(1,2)

        # return out[:,:,:self.length_output]
        return decoder_output[:,:,:self.length_output]

class Postnet(nn.Module):
    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(80, 512,
                         kernel_size=5, stride=1, dilation=1,
                         padding=2, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
        )
        for i in range(4):
            self.convolutions.append(
                nn.Sequential(
                ConvNorm(512, 512,
                         kernel_size=5, stride=1, dilation=1,
                         padding=2, w_init_gain='tanh'),
                nn.BatchNorm1d(512))
            )
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(512, 80,
                        kernel_size=5, stride=1, padding=2, dilation=1,
                        w_init_gain='linear'),
                nn.BatchNorm1d(80))
        )
        self.apply(init_weights)

    # def forward(self, x):
    #     for i in range(len(self.convolutions)-1):
    #         x = torch.tanh(self.convolutions[i](x))
    #     x = torch.sigmoid(self.convolutions[-1](x))
    #     return x

    def forward(self, x):
        for i in range(len(self.convolutions)-1):
            x = torch.tanh(self.convolutions[i](x))
        x = torch.tanh(self.convolutions[-1](x))
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, num_speaker):
        super(Classifier, self).__init__()

        self.linear1 = nn.Linear(input_dim//2, 256)
        self.linear2 = nn.Linear(256, num_speaker)
        # self.softmax = nn.sof
        self.apply(init_weights)

    def forward(self, z):
        out = torch.relu(self.linear1(z))
        out = torch.relu(self.linear2(out))
        # out = torch.nn.functional.softmax(out)
        return out

####################################################################################
# encoder = Encoder(128, 32,80).to(device)
# decoder = Decoder(128, 80, 131).to(device)
# posnet = Postnet().to(device)
# classifier = Classifier(64, 109).to(device)
# data = torch.rand(5, 80, 131).to(device)

# mu1, logvar1, mu2, logvar2 = encoder(data)
# reconstruct = decoder(mu1, logvar1, mu2, logvar2)
# labels = classifier(mu1)
# post_mel = posnet(reconstruct)
# max_prob = torch.max(labels, dim=1)
# print(reconstruct.shape)
# print(post_mel.shape)
# print(labels.shape)
# print(max_prob)

# print(index_to_onehot(2))
