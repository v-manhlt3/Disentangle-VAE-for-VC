import sys
sys.path.append('..')

import torch
import torch.nn as nn
import numpy as np
from time import time
import os
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

from plot import artificial_data_reconstruction_plot, emnist_plot_samples, emnist_plot_spectrum, emnist_plot_variation_along_dims

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from FrEIA.framework import *
from FrEIA.modules import *
from preprocessing.dataset_mel import SpeechDatasetMCC, SpeechDataset2

class GIN(nn.Module):
    def __init__(self, dataset_fp, n_epochs, epochs_per_line, lr,
                lr_schedule, batch_size, save_frequency, incompressible_flow,
                empirical_vars, data_root_dir='./', n_classes=None, n_data_points=None, init_identity=True):
        super().__init__()
        self.timestamp = str(int(time()))
        self.save_dir = os.path.join('/home/ubuntu/VCC2016_dataset_logs', self.timestamp)
        # self.dataset = dataset
        self.n_epochs = n_epochs
        self.epochs_per_line = epochs_per_line
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.batch_size = batch_size
        self.save_frequency = min(save_frequency, n_epochs)
        self.incompressible_flow = bool(incompressible_flow)
        self.empirical_vars = bool(empirical_vars)
        self.init_identity = bool(init_identity)
        self.n_classes = 10
        self.n_dims = 80*16
        # self.net = construct_net_emnist(coupling_block='gin' if self.incompressible_flow else 'glow')
        self.net = construct_net_emnist(coupling_block='glow')
        # self.net = build_inn()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.dataset = SpeechDataset2(dataset_fp, samples_length=16)
        self.train_loader = DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        self.test_loader = DataLoader(self.dataset, batch_size=40, pin_memory=True, shuffle=True)

        self.to(self.device)
            
    def forward(self, x, rev=False):
        x = self.net(x, rev=rev)
        return x
    
    def train_model(self):
        os.makedirs(self.save_dir)
        with open(os.path.join(self.save_dir, 'log.txt'), 'w') as f:
            f.write(f'incompressible_flow {self.incompressible_flow}\n')
            f.write(f'empirical_vars {self.empirical_vars}\n')
            f.write(f'init_identity {self.init_identity}\n')
        os.makedirs(os.path.join(self.save_dir, 'model_save'))
        os.makedirs(os.path.join(self.save_dir, 'figures'))
        print(f'\nTraining model for {self.n_epochs} epochs \n')
        self.train()
        self.to(self.device)
        print('  time     epoch    iteration         loss       last checkpoint')
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.lr_schedule)
        losses = []
        t0 = time()
        for epoch in range(self.n_epochs):
            self.epoch = epoch
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if self.empirical_vars:
                    # first check that std will be well defined
                    if min([sum(target==i).item() for i in range(self.n_classes)]) < 2:
                        # print('skip training due to arthemic issue')
                        # don't calculate loss and update weights -- it will give nan or error
                        # go to next batch
                        continue
                optimizer.zero_grad()
                data = data.unsqueeze(1).float()
                data += torch.randn_like(data)*1e-2
                data = data.to(self.device)
                z = self.net(data)          # latent space variable
                # z = self.net(data, c=one_hot(target))
                # print('this is shape of latent tensor: ', z.shape)
                # print('target label', target[10])
                logdet_J = self.net.log_jacobian(run_forward=False)
                if self.empirical_vars:
                    # we only need to calculate the std
                    sig = torch.stack([z[target==i].std(0, unbiased=False) for i in range(self.n_classes)])
                    # negative log-likelihood for gaussian in latent space
                    loss = 0.5 + sig[target].log().mean(1) + 0.5*np.log(2*np.pi)
                else:
                    m = self.mu[target]
                    ls = self.log_sig[target]
                    # negative log-likelihood for gaussian in latent space
                    loss = torch.mean(0.5*(z-m)**2 * torch.exp(-2*ls) + ls, 1) + 0.5*np.log(2*np.pi)
                loss -= logdet_J / self.n_dims
                loss = loss.mean()
                self.print_loss(loss.item(), batch_idx, epoch, t0)
                losses.append(loss.item())
                loss.backward(retain_graph=True)
                optimizer.step()
            if (epoch+1)%self.epochs_per_line == 0:
                avg_loss = np.mean(losses)
                self.print_loss(avg_loss, batch_idx, epoch, t0, new_line=True)
                losses = []
            sched.step()
            if (epoch+1)%self.save_frequency == 0:
                self.save(os.path.join(self.save_dir, 'model_save', f'{epoch+1:03d}.pt'))
                self.make_plots()
    
    def print_loss(self, loss, batch_idx, epoch, t0, new_line=False):
        n_batches = len(self.train_loader)
        print_str = f'  {(time()-t0)/60:5.1f}   {epoch+1:03d}/{self.n_epochs:03d}   {batch_idx+1:04d}/{n_batches:04d}   {loss:12.4f}'
        if new_line:
            print(print_str+' '*40)
        else:
            last_save = (epoch//self.save_frequency)*self.save_frequency
            if last_save != 0:
                print_str += f'           {last_save:03d}'
            print(print_str, end='\r')
    
    def save(self, fname):
        state_dict = OrderedDict((k,v) for k,v in self.state_dict().items() if not k.startswith('net.tmp_var'))
        torch.save({'model': state_dict}, fname)
    
    def load(self, fname):
        data = torch.load(fname)
        self.load_state_dict(data['model'])
    
    def make_plots(self):
        # if self.dataset == '10d':
        #     artificial_data_reconstruction_plot(self, self.latent, self.data, self.target)
        # elif self.dataset == 'EMNIST':
        os.makedirs(os.path.join(self.save_dir, 'figures', f'epoch_{self.epoch+1:03d}'))
        self.set_mu_sig()
        sig_rms = np.sqrt(np.mean((self.sig**2).detach().cpu().numpy(), axis=0))
        emnist_plot_samples(self, n_rows=3)
        emnist_plot_spectrum(self, sig_rms)
        n_dims_to_plot = 100
        top_sig_dims = np.flip(np.argsort(sig_rms))
        dims_to_plot = top_sig_dims[:n_dims_to_plot]
        emnist_plot_variation_along_dims(self, dims_to_plot)
        # else:
        #     raise RuntimeError("Check dataset name. Doesn't match.")
    def set_mu_sig(self, init=False, n_batches=40):
        if self.empirical_vars or init:
            examples = iter(self.test_loader)
            # examples =  examples.unsqueeze(1).float()
            n_batches = min(n_batches, len(examples))
            latent = []
            target = []
            for _ in range(n_batches):
                data, targ = next(examples)
                data =  data.unsqueeze(1).float()
                data += torch.randn_like(data)*1e-2
                self.eval()
                latent.append(self(data.to(self.device)).detach().cpu())
                target.append(targ)
            latent = torch.cat(latent, 0)
            target = torch.cat(target, 0)
        if self.empirical_vars:
            self.mu = torch.stack([latent[target == i].mean(0) for i in range(10)]).to(self.device)
            self.sig = torch.stack([latent[target == i].std(0) for i in range(10)]).to(self.device)
        else:
            if init:
                self.mu.data = torch.stack([latent[target == i].mean(0) for i in range(10)])
                self.log_sig.data = torch.stack([latent[target == i].std(0) for i in range(10)]).log()
            else:
                self.sig = self.log_sig.exp().detach()



def subnet_fc_10d(c_in, c_out, init_identity):
    subnet = nn.Sequential(nn.Linear(c_in, 10), nn.ReLU(),
                            nn.Linear(10, 10), nn.ReLU(),
                            nn.Linear(10,  c_out))
    if init_identity:
        subnet[-1].weight.data.fill_(0.)
        subnet[-1].bias.data.fill_(0.)
    return subnet


def construct_net_10d(coupling_block, init_identity=True):
    if coupling_block == 'gin':
        block = Fm.GINCouplingBlock
    else:
        assert coupling_block == 'glow'
        block = Fm.GLOWCouplingBlock
    
    nodes = [Ff.InputNode(10, name='input')]
    
    for k in range(8):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':lambda c_in,c_out: subnet_fc_10d(c_in, c_out, init_identity), 'clamp':2.0},
                             name=F'coupling_{k}'))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom,
                        {'seed':np.random.randint(2**31)},
                        name=F'permute_{k+1}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    return Ff.ReversibleGraphNet(nodes)


def subnet_fc(c_in, c_out):
    width = 1024
    subnet = nn.Sequential(nn.Linear(c_in, width), nn.ReLU(),
                           nn.Linear(width, width), nn.ReLU(),
                           nn.Linear(width,  c_out))
    for l in subnet:
        if isinstance(l, nn.Linear):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def subnet_conv1(c_in, c_out):
    width = 32
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def subnet_conv2(c_in, c_out):
    width = 64
    subnet = nn.Sequential(nn.Conv2d(c_in, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(width, c_out, 3, padding=1))
    for l in subnet:
        if isinstance(l, nn.Conv2d):
            nn.init.xavier_normal_(l.weight)
    subnet[-1].weight.data.fill_(0.)
    subnet[-1].bias.data.fill_(0.)
    return subnet

def construct_net_emnist(coupling_block):
    if coupling_block == 'gin':
        block = Fm.GINCouplingBlock
    else:
        assert coupling_block == 'glow'
        block = Fm.GLOWCouplingBlock
    
    nodes = [Ff.InputNode(1, 16, 80, name='input')]
    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name='downsample1'))

    for k in range(16):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':subnet_conv1, 'clamp':2.0},
                             name=F'coupling_conv1_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':np.random.randint(2**31)},
                             name=F'permute_conv1_{k}'))

    nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name='downsample2'))

    for k in range(16):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':subnet_conv2, 'clamp':2.0},
                             name=F'coupling_conv2_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':np.random.randint(2**31)},
                             name=F'permute_conv2_{k}'))

    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

    for k in range(8):
        nodes.append(Ff.Node(nodes[-1], block,
                             {'subnet_constructor':subnet_fc, 'clamp':2.0},
                             name=F'coupling_fc_{k}'))
        nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':np.random.randint(2**31)},
                             name=F'permute_fc_{k}'))

    nodes.append(Ff.OutputNode(nodes[-1], name='output'))
    return Ff.ReversibleGraphNet(nodes)




# function is here rather than in data.py to prevent circular import
def generate_artificial_data_10d(n_clusters, n_data_points):
    latent_means = torch.rand(n_clusters, 2)*10 - 5         # in range (-5, 5)
    latent_stds  = torch.rand(n_clusters, 2)*2.5 + 0.5      # in range (0.5, 3)
    
    labels = torch.randint(n_clusters, size=(n_data_points,))
    latent = latent_means[labels] + torch.randn(n_data_points, 2)*latent_stds[labels]
    latent = torch.cat([latent, torch.randn(n_data_points, 8)*1e-2], 1)
    
    random_transf = construct_net_10d('glow', init_identity=False)
    data = random_transf(latent).detach()
    
    return latent, data, labels

def build_inn():

        def sub_conv(ch_hidden, kernel):
            pad = kernel // 2
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
                                            nn.ReLU(),
                                            nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad))

        def sub_fc(ch_hidden):
            return lambda ch_in, ch_out: nn.Sequential(
                                            nn.Linear(ch_in, ch_hidden),
                                            nn.ReLU(),
                                            nn.Linear(ch_hidden, ch_out))

        nodes = [Ff.InputNode(1, 16, 80)]
        cond_size = 10
        # cond_node1 = Ff.ConditionNode(10, 8, 40)
        # cond_node2 = Ff.ConditionNode(10, 4, 20)
        # cond_node3 = Ff.ConditionNode(10, 2, 10)
        # cond_node4 = Ff.ConditionNode(160)
        # outputs of the cond. net at different resolution levels
        conditions = [Ff.ConditionNode(4, 8, 40),
                      Ff.ConditionNode(16, 4, 20),
                      Ff.ConditionNode(54, 2, 10),
                      Ff.ConditionNode(160)]

        split_nodes = []

        subnet = sub_conv(32,3)
        nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}, name='downsample1'))
        # nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}))
        for k in range(2):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0}, conditions=conditions[0],name='aaa'+str(k)))
            nodes.append(Ff.Node(nodes[-1],
                             Fm.PermuteRandom,
                             {'seed':np.random.randint(2**31)},
                             name=F'permute_conv1_{k}'))
            

        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}))

        for k in range(4):
            subnet = sub_conv(64, 3 if k%2 else 1)

            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0}, conditions=conditions[1]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))

        #split off 6/8 ch
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections':[4,12], 'dim':0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))

        nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}))

        for k in range(4):
            subnet = sub_conv(128, 3 if k%2 else 1)

            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':0.6}, conditions=conditions[2]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))

        #split off 4/8 ch
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections':[8,8], 'dim':0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

        # fully_connected part
        subnet = sub_fc(512)
        for k in range(4):
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':0.6}, conditions=conditions[3]))
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))

        # concat everything
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                             Fm.Concat1d, {'dim':0}))
        nodes.append(Ff.OutputNode(nodes[-1]))

        return Ff.ReversibleGraphNet(nodes +conditions, verbose=False)

def one_hot(labels, out=None):
    '''
    Convert LongTensor labels (contains labels 0-9), to a one hot vector.
    Can be done in-place using the out-argument (faster, re-use of GPU memory)
    '''
    if out is None:
        out = torch.zeros(labels.shape[0], 10).to(labels.device)
    else:
        out.zeros_()

    out.scatter_(dim=1, index=labels.view(-1,1), value=1.)
    return out
