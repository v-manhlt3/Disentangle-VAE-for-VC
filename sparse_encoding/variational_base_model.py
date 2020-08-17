import torch
from torchvision.utils import save_image
from pathlib import Path
from glob import glob
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import librosa.display
from sparse_encoding.logger import Logger
from sparse_encoding.plot import *
from tensorboardX import SummaryWriter
from preprocessing.processing import build_model, wavegen
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VariationalBaseModel():
    def __init__(self, dataset, width, height, channels, latent_sz, 
                 learning_rate, device, log_interval, batch_size, normalize=False, 
                 flatten=True):
        self.dataset = dataset
        self.width = width
        self.height = height
        self.channels = channels
        # before width * height * channels
        self.input_sz = (channels, width, height)
        self.latent_sz = latent_sz
        
        self.lr = learning_rate
        self.device = device
        self.log_interval = log_interval
        self.normalize_data = normalize
        self.flatten_data = flatten
        
        # To be implemented by subclasses
        self.model = None
        self.optimizer = None
        self.batch_size = batch_size        
    
    
    def loss_function(self):
        raise NotImplementedError
    
    
    def step(self, data, train=False):
        if train:
            self.optimizer.zero_grad()
        shape = data.shape
        # data = data.view(shape[0]*shape[1], shape[2], shape[3])
        output0, output,mu,logvar,logspike = self.model(data)
        output =  torch.clamp(output, min=0, max=1.0)

        loss, recons_loss0, recons_loss, prior_loss = self.loss_function(data, output0, output,mu, logvar,logspike, train=train)
        if train:
            loss.backward()
            self.optimizer.step()
        return loss.item(), recons_loss.item(), prior_loss.item()
    
    # TODO: Perform transformations inside DataLoader (extend datasets.MNIST)
    def transform(self, batch):
        if self.flatten_data: 
            batch_size = len(batch)
            batch = batch.view(batch_size, -1)
        if self.normalize_data:
            batch = batch / self.scaling_factor
#         batch_norm = flattened_batch.norm(dim=1, p=2)
#         flattened_batch /= batch_norm[:, None]
        return batch
        
    def inverse_transform(self, batch):
        return batch * self.scaling_factor \
                if self.normalize_data else batch
    
    def calculate_scaling_factor(self, data_loader):
        print(f'Calculating norm mean of training set')
        norms = []
        self.model.eval()
        n_batches = len(data_loader)
        for batch_idx, (data, _,_) in enumerate(data_loader):
            batch_size = len(data)
            flattened_batch = data.view(batch_size, -1)
            batch_norm = flattened_batch.norm(dim=1, p=2)
            norms.extend(list(batch_norm.numpy()))
        norms = pd.Series(norms)
        print(norms.describe())
        self.scaling_factor = norms.mean()
        print('Done!\n')
    
    
    # Run training iterations and report results
    def train(self, train_loader, epoch, logging_func=print):
        self.model.train()
        train_loss = 0
        total_recons_loss, total_prior_loss = 0, 0
        for batch_idx, (data, _,_) in enumerate(train_loader):

            data = data.to(torch.device("cuda")).float()
            loss, recons_loss, prior_loss = self.step(data, train=True)
            train_loss += loss
            total_prior_loss += prior_loss
            total_recons_loss += recons_loss

            if (batch_idx+1) % self.log_interval == 0:
                logging_func('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}' \
                      .format(epoch, batch_idx * len(data), 
                              len(train_loader.dataset),
                              100. * batch_idx / len(train_loader),
                              loss / len(data)))

        logging_func('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
        
        return total_recons_loss, total_prior_loss
        
        
    # Returns the VLB for the test set
    def test(self, test_loader, epoch, logging_func=print):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, _,_ in test_loader:
                # data = self.transform(data).to(self.device)
                data = data.to(torch.device("cuda")).float()
                test_loss,_,_ = self.step(data, train=False)
                
        VLB = test_loss / len(test_loader)
        ## Optional to normalize VLB on testset
        name = self.model.__class__.__name__
        test_loss /= len(test_loader.dataset) 
        logging_func(f'====> Test set loss: {test_loss:.4f} - VLB-{name} : {VLB:.4f}')
        return test_loss
    
    
    #Auxiliary function to continue training from last trained models
    def load_last_model(self, checkpoints_path, logging_func=print):
        name = self.model.__class__.__name__
        # Search for all previous checkpoints
        models = glob(f'{checkpoints_path}/*.pth')
        model_ids = []
        for f in models:
            # modelname_dataset_startepoch_epochs_latentsize_lr_epoch
            run_name = Path(f).stem
            model_name, dataset, _, _, latent_sz, _, epoch = run_name.split('_')
            print('-------current epoch: ', epoch)
            # if model_name == name and dataset == self.dataset and \
            #    int(latent_sz) == self.latent_sz:
            model_ids.append((int(epoch), f))
                
        # If no checkpoints available
        
        if len(model_ids) == 0:
            print('model_ids: ', model_ids)
            logging_func(f'Training {name} model from scratch...')
            return 1

        # Load model from last checkpoint 
        start_epoch, last_checkpoint = max(model_ids, key=lambda item: item[0])
        logging_func('Last checkpoint: ', last_checkpoint)
        self.model.load_state_dict(torch.load(last_checkpoint))
        logging_func(f'Loading {name} model from last checkpoint ({start_epoch})...')

        return start_epoch + 1
    
    
    def update_(self):
        pass
    
    
    def run_training(self, train_loader, test_loader, epochs, 
                     report_interval, sample_sz=64, reload_model=True,
                     checkpoints_path='../results2/checkpoints',
                     logs_path='../results2/logs',
                     images_path='../results2/images',
                     estimation_dir='../results2/images/estimation',
                     logging_func=print, start_epoch=None):
        
        if self.normalize_data:
            self.calculate_scaling_factor(train_loader)
        
        if reload_model:
            start_epoch = self.load_last_model(checkpoints_path, logging_func)
        else:
            start_epoch = 1
        name = self.model.__class__.__name__
        run_name = f'{name}_{self.dataset}_{start_epoch}_{epochs}_' \
                   f'{self.latent_sz}_{str(self.lr).replace(".", "-")}'
        # logger = Logger(f'{logs_path}/{run_name}')
        # logging_func(f'Training {name} model...')
        writer = SummaryWriter(f'{logs_path}/{run_name}')
        for epoch in range(start_epoch, start_epoch + epochs):
            total_recons_loss, total_prior_loss = self.train(train_loader, epoch, logging_func)
            test_loss = self.test(test_loader, epoch, logging_func)
            total_recons_loss /= len(train_loader)
            total_prior_loss /= len(train_loader)
            # logger.scalar_summary(train_loss, test_loss, epoch)
            # logger.write_log(total_recons_loss, total_prior_loss, epoch)
            print('recons loss epoch_{}: {}'.format(epoch, total_recons_loss / (len(train_loader)*self.batch_size)))
            print('prior loss epoch_{}: {}'.format(epoch, total_prior_loss / (len(train_loader)*self.batch_size)))
            writer.add_scalar('Loss\Reconstruction Loss', total_recons_loss / (len(train_loader)*self.batch_size), epoch)
            writer.add_scalar('Loss\Prior Loss', total_prior_loss / (len(train_loader)*self.batch_size), epoch)
            # Optional update
            self.update_()
            # For each report interval store model and save images
            if epoch % report_interval == 0:
                if not os.path.exists(images_path):
                    os.mkdir(images_path)
                if not os.path.exists(checkpoints_path):
                    os.mkdir(checkpoints_path)
                with torch.no_grad():
                    ## Generate random samples
                    # sample = torch.randn(sample_sz, self.latent_sz) \
                    #               .to(self.device)
                    # sample = self.model.decode(sample).cpu()
                    # sample = self.inverse_transform(sample)
                    # ## Store sample plots
                    # save_image(sample.view(sample_sz, self.channels, self.height,
                    #                        self.width),
                    #            f'{images_path}/sample_{run_name}_{epoch}.png')
                    ## Store Model
                    torch.save(self.model.state_dict(), 
                               f'{checkpoints_path}/{run_name}_{epoch}.pth')
                    self.estimate_trained_model(test_loader, checkpoints_path, estimation_dir)

    def estimate_trained_model(self, test_loader, checkpoints_path, estimation_dir):

        logging_epoch = self.load_last_model(checkpoints_path, logging_func=print)
        self.model.eval()
        # estimate_dir = '../results2/images/estimation'
        if not os.path.exists(estimation_dir):
            os.mkdir(estimation_dir)

        with torch.no_grad():
            data,_,_ = next(iter(test_loader))
            shape = data.shape
            # data = data.view(shape[0]*shape[1], shape[2], shape[3])
            data = data.to(torch.device("cuda")).float()
            recons_data0, recons_data, mu, logvar, logspike = self.model(data, train=False)
            
            for i in range(5):

                original_mel_fp = os.path.join(estimation_dir, str(logging_epoch) + '_original_mel_' +str(i)+'.png')
                recons_mel_fp = os.path.join(estimation_dir, str(logging_epoch) + '_recons_mel_' +str(i)+'.png')
                recons_mel = recons_data[i]
                origin_mel = data[i]
                # z = self.model.reparameterize(mu[i], logvar[i], logspike[i])
                z = mu[i]

                plt.figure()
                plt.title('reconstructed mel spectrogram')
                librosa.display.specshow(recons_mel.cpu().detach().numpy(), x_axis='time', y_axis='mel', sr=16000)
                plt.colorbar(format='%f')
                plt.savefig(recons_mel_fp)

                plt.figure()
                plt.title('original mel spectrogram')
                librosa.display.specshow(origin_mel.cpu().detach().numpy(), x_axis='time', y_axis='mel', sr=16000)
                plt.colorbar(format='%f')
                plt.savefig(original_mel_fp)

                encoding_visualization(z, estimation_dir, i, logging_epoch)


    def generate_wav(self, test_loader, ckp_path, generation_dir):
        
        origin_mel_fp = '/home/ubuntu/vcc2016_train/vcc2016_training_SM1/100001.npy'

        epoch = self.load_last_model(ckp_path, logging_func=print)
        self.model.eval()
        vocoder_model = build_model().to(device)
        ckpt = torch.load('/home/ubuntu/checkpoint_step001000000_ema.pth')
        vocoder_model.load_state_dict(ckpt['state_dict'])
        if not os.path.exists(generation_dir):
            os.mkdir(generation_dir)

        with torch.no_grad():
            # data, utterances, speakers = next(iter(test_loader))
            # data = data.to(torch.device("cuda")).float()
            original_wav_fn = 'SM1_100001'
            
            print('=====>the original wav is: ', original_wav_fn)
            original_mel = np.load(origin_mel_fp)
            print('melspectrogram shape: ', original_mel.shape)
            data = chunking_mel(original_mel)
            print('data shape: ', data.shape)
            data = data.to(torch.device("cuda")).float()
            _, recons_data, _, _, _ = self.model(data, train=False)

            # original_mel = data[0].cpu().detach().numpy()
            recons_mel = torch.cat([recons_data[i] for i in range(recons_data.shape[0])], 1)
            print('recons mel shape: ', recons_mel.shape)
            recons_mel = recons_mel.cpu().detach().numpy()

            plt.figure()
            plt.title(original_wav_fn)
            librosa.display.specshow(original_mel, x_axis='time', y_axis='mel', sr=16000)
            plt.colorbar(format='%f')
            plt.savefig(os.path.join(generation_dir, original_wav_fn + '.png'))

            plt.figure()
            plt.title('reconstruct_'+original_wav_fn)
            librosa.display.specshow(recons_mel, x_axis='time', y_axis='mel', sr=16000)
            plt.colorbar(format='%f')
            plt.savefig(os.path.join(generation_dir, 'recons_'+original_wav_fn + '.png'))

            recons_mel = np.transpose(recons_mel, (-1, -2))
            waveform = wavegen(vocoder_model, recons_mel)
            librosa.output.write_wav(os.path.join(generation_dir, 'recons_'+original_wav_fn + '.wav'), waveform, sr=16000)
    
    def analyze_latent_code(self, speaker_id, estimation_dir, ckp_path, dataset, utterance=None):

        epoch = self.load_last_model(ckp_path, logging_func=print)
        self.model.eval()
        if utterance == None:
            batch_data = dataset.get_batch_utterances(speaker_id, 100)
            batch_data =  batch_data.cuda().float()
        else:
            batch_data,_,_ = dataset.get_batch_speaker(utterance)
            batch_data =  batch_data.cuda().float()
            speaker_id = 'analysi_utt' + utterance.split('.')[0]
        _,_,latent_vectors,_,_ = self.model(batch_data)

        plot_latentvt_analysis(latent_vectors, estimation_dir, speaker_id, threshold_mean=0.1, threshold_std=0.2)


    def voice_conversion(self, target_speaker, source_speaker,
                        utterance_id, dataset, ckp_path, generation_dir):

        epoch = self.load_last_model(ckp_path, logging_func=print)
        self.model.eval()
        vocoder_model = build_model().to(device)
        ckpt = torch.load('/home/ubuntu/checkpoint_step001000000_ema.pth')
        vocoder_model.load_state_dict(ckpt['state_dict'])

        source_data = dataset.get_batch_utterances(source_speaker, 30).cuda().float()
        target_data = dataset.get_batch_utterances(target_speaker, 30).cuda().float()
        utterance = dataset.get_utterance(source_speaker, utterance_id)
        utterance = chunking_mel(utterance).cuda().float()

        with torch.no_grad():
            _,_, source_latent_vectors,_,_ = self.model(source_data)
            _,_, target_latent_vectors,_,_ = self.model(target_data)
            
            source_idx, source_mean, source_std = plot_latentvt_analysis(source_latent_vectors,
                                                                            generation_dir, source_speaker, 0.1, 0.7)
            target_idx, target_mean, target_std = plot_latentvt_analysis(target_latent_vectors,
                                                                            generation_dir, target_speaker, 0.1, 0.7)
            target_mean = torch.from_numpy(target_mean)
            source_mean = torch.from_numpy(source_mean)
            mu, logvar, logspike = self.model.encode(utterance)
            backup_mu = mu.clone()
            print('target mean: ', target_mean)
            print('target index: ', len(target_idx))
            print('source index: ', len(source_idx))

            for j in range(mu.shape[0]):
                for i, idx in enumerate(source_idx):
                    mu[j, idx] = 0

            for j in range(mu.shape[0]):    
                for i, idx in enumerate(source_idx):
                    mu[j,idx] = source_mean[i]

            recons_mel = self.model.decode(backup_mu)
            recons_voice = torch.cat([recons_mel[i] for i in range(recons_mel.shape[0])], 1)
            recons_voice = recons_voice.cpu().detach().numpy()
            # converted_voice = np.transpose(converted_voice, (-1, -2))

            converted_mel = self.model.decode(mu)
            converted_voice = torch.cat([converted_mel[i] for i in range(converted_mel.shape[0])], 1)
            converted_voice = converted_voice.cpu().detach().numpy()

            plt.figure()
            plt.title('convert_' + source_speaker +'_'+ target_speaker+'_'+ utterance_id)
            librosa.display.specshow(converted_voice, x_axis='time', y_axis='mel', sr=16000)
            plt.colorbar(format='%f')
            plt.savefig(os.path.join(generation_dir, 'convert_' + source_speaker +'_'+ target_speaker+'_'+ utterance_id+'.png'))

            plt.figure()
            plt.title('reconstruct_' + source_speaker +'_' + utterance_id)
            librosa.display.specshow(recons_voice, x_axis='time', y_axis='mel', sr=16000)
            plt.colorbar(format='%f')
            plt.savefig(os.path.join(generation_dir, 'recons_'+ source_speaker +'_' + utterance_id + '.png'))
            
            converted_voice = np.transpose(converted_voice, (-1, -2))
            waveform = wavegen(vocoder_model, converted_voice)
            librosa.output.write_wav(os.path.join(generation_dir,
            'convert_'+source_speaker+'_to_'+target_speaker+'_'+utterance_id.split('.')[0]+'.wav'), waveform, sr=16000)

            



############################################# Helper functions #####################################
def chunking_mel(melspectrogram):
    data = []
    num_spectro = (melspectrogram.shape[1]//64)+1
    print('num_spectro: ', num_spectro)
    for index in range(num_spectro):
        if index < num_spectro - 1:
            mel = melspectrogram[:,index*64:index*64+64]
            print('mel: ', mel.shape)
        else:
            mel = melspectrogram[:,index*64:]
            mel = np.pad(mel, ((0,0),(0, 64 - melspectrogram.shape[1]%64)), 'constant', constant_values=(0,0))
            print('last mel shape: ', mel.shape)
        data.append(mel)
    return torch.tensor(data)

