import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as numpy
import librosa
from autovc import init_weights
from autovc import ConvNorm

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

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
            # nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            ConvNorm(512, 512, kernel_size=5, stride=2, padding=2, w_init_gain='relu'),
            nn.BatchNorm1d(512),
            # nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            ConvNorm(512, 512, kernel_size=5, stride=2, padding=2, w_init_gain='relu'),
            nn.BatchNorm1d(512),
            # nn.ReLU()
        )

        self.lstm = nn.LSTM(512, dim_neck, 2, batch_first=True, bidirectional=True)

        self.mu = nn.Linear(33*dim_neck*2, latent_dim//2)
        self.logvar = nn.Linear(33*dim_neck*2, latent_dim//2)

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

        mu = self.mu(outputs)
        logvar = self.logvar(outputs)

        
        # return codes
        return mu, logvar

class Model(nn.Module):
    def __init__(self, input_dims=10, simi_matrix_size=20):
        super(Model, self).__init__()

        self.input_dims = input_dims
        self.simi_matrix_size = simi_matrix_size
        self.cos_simi_layer = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        self.w = nn.Parameter(torch.randn(1), requires_grad=True)
        self.b = nn.Parameter(torch.randn(1), requires_grad=True)

        self.apply(init_weights)

    def forward(self, speaker_encode, centroids):
        torch.clamp_min(self.w, 1e-6)
        simi_matrix = []
        for centroid in centroids:
            cosine = self.cos_simi_layer(speaker_encode, centroid.unsqueeze(dim=0))
            similarity = self.w*cosine + self.b
            simi_matrix.append(similarity)
        simi_matrix = torch.stack(simi_matrix, dim=1)
        return simi_matrix.transpose_(1, 0)

def get_centroids(num_speaker, num_utterance, mini_batch, current_sample):
    centroids = []
    current_speaker = current_sample % num_utterance
    for i in range(num_speaker):
        if i == current_speaker:
            batch = torch.cat((mini_batch[i*num_utterance:current_sample], mini_batch[current_sample+1:i*num_utterance+num_utterance]), dim=0)
        else:
            batch = mini_batch[i*num_utterance:i*num_utterance+num_utterance]
        centroid = torch.mean(batch, dim=0)
        centroids.append(centroid)
    return torch.stack(centroids, dim=1).transpose_(1, 0)

def calc_loss(simi_matrix):
    total_loss = 0
    # print('simi matrix shape: ', simi_matrix.shape)
    for i in range(simi_matrix.shape[0]):
        row_loss = 0
        for j in range(simi_matrix.shape[1]):
            if i==j:
                continue
                # row_loss += (-1.0)*simi_matrix[j][j]
            else:
                row_loss = row_loss + torch.exp(simi_matrix[i][j])
        row_loss = torch.log(row_loss) + simi_matrix[i][i]
        total_loss = total_loss + row_loss
    return total_loss

######################################################################################
@timing
def main():
    device = torch.device('cuda')
    dataset_path = '/home/ubuntu/VCTK-Corpus/wav16/'
    from preprocessing.dataset_mel import SpeechDataset
    from torch.utils.data import DataLoader
    dataset = SpeechDataset(dataset_path)
    dataloader = DataLoader(dataset, num_workers=8, batch_size=2,
                            pin_memory=True, shuffle=True, drop_last=True)

    # data,_,_ = next(iter(dataloader))
    


    encoder = Encoder(128, 32, 80).to(device)
    model = Model(10, 20).to(device)
    for data, utterances, speaker_id in iter(dataloader):
        data = data.view(-1, 80, 131).to(device).float()
        mu, logvar = encoder(data)
        total_loss = 0 
        for idx in range(mu.shape[0] - 1):
            centroids = get_centroids(2, 10, mu, idx)
            simi_matrix = model(mu, centroids)
            loss = calc_loss(simi_matrix)
            total_loss += loss
    
    print('This is total loss: ', total_loss)
if __name__=='__main__':
    
    main()
    # output = model(a,b)
    # print(output)
    # print(output)