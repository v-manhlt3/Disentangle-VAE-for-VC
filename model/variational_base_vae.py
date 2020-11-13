import torch
from torchvision.utils import save_image
from pathlib import Path
from glob import glob
import pandas as pd
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import librosa.display
from model.logger import Logger
from model.plot import *
from model.train_feature_selection import feature_selection
from model.feature_selection import FeatureSelection
from model import utils
from model import disentangled_vae

from tqdm import tqdm
from tensorboardX import SummaryWriter
from preprocessing.processing import build_model, wavegen
from preprocessing.WORLD_processing import *
import numpy as np
import librosa




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VariationalBaseModelVAE():
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
    
    
    def step(self, data1, data2, speaker_ids, train=False):
        if train:
            self.optimizer.zero_grad()

        recons_x1, recons_x2, recons_x1_hat,recons_x2_hat,q_z1_mu, q_z1_logvar, q_z2_mu, q_z2_logvar, style_mu, style_logvar =\
        self.model(data1, data2)

        loss, recons_loss1, recons_loss2, recons_loss1_hat, recons_loss2_hat, z1_kl_loss, z2_kl_loss, z_style_kl = \
        self.loss_functionGVAE2(data1,data2,recons_x1, recons_x2, recons_x1_hat, recons_x2_hat,q_z1_mu,q_z1_logvar,q_z2_mu, q_z2_logvar,style_mu, style_logvar,train=train)
        if train:
            loss.backward()
            self.optimizer.step()
        return loss.item(), recons_loss1.item(), recons_loss2.item(), recons_loss1_hat.item(), recons_loss2_hat.item(), z1_kl_loss.item(), z2_kl_loss.item(), z_style_kl.item()

    
    # Run training iterations and report results
    def train(self, train_loader, epoch, logging_func=print):
        self.model.train()
        train_loss, total_z_style_kl = 0, 0
        total_recons_loss1, total_recons_loss2, total_z1_kl_loss, total_z2_kl_loss = 0, 0, 0, 0
        total_recons_loss1_hat, total_recons_loss2_hat = 0,0
        for batch_idx, (data1, data2, speaker_ids) in enumerate(tqdm(train_loader)):
            
            data1 = data1.to(torch.device("cuda")).float()
            data2 = data2.to(torch.device("cuda")).float()
            speaker_ids = speaker_ids.view(-1)


            loss, recons_loss1, recons_loss2,recons_loss1_hat, recons_loss2_hat, z1_kl_loss, z2_kl_loss, z_style_kl = self.step(data1, data2, speaker_ids,train=True)
            train_loss += loss
            total_recons_loss1 += recons_loss1
            total_recons_loss2 += recons_loss2
            total_z1_kl_loss += z1_kl_loss
            total_z2_kl_loss += z2_kl_loss
            total_z_style_kl += z_style_kl
            total_recons_loss1_hat += recons_loss1_hat
            total_recons_loss2_hat += recons_loss2_hat

        train_loader.dataset.shuffle_data()

        logging_func('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
        
        return total_recons_loss1, total_recons_loss2, total_recons_loss1_hat, total_recons_loss2_hat, total_z1_kl_loss, total_z2_kl_loss, z_style_kl
        
        
    # Returns the VLB for the test set
    def test(self, test_loader, epoch, logging_func=print):
        self.model.eval()
        test_loss = 0
        total_recons_loss = 0
        with torch.no_grad():
            for data, _,speaker_ids in test_loader:
                # data = self.transform(data).to(self.device)
                data = data.to(torch.device("cuda")).float()
                data = data.view(-1, data.shape[-2], data.shape[-1])
                speaker_ids = speaker_ids.view(-1)
                test_loss, recons_loss,_,_ = self.step(data, speaker_ids,train=False)
                total_recons_loss += recons_loss
        VLB = test_loss / len(test_loader)
        ## Optional to normalize VLB on testset
        name = self.model.__class__.__name__
        test_loss /= len(test_loader.dataset)
        average_recons_loss = (total_recons_loss/len(test_loader)) 
        logging_func(f'====> Test recons loss: {average_recons_loss:.4f} - VLB-{name} : {VLB:.4f}')
        return test_loss
    
    
    #Auxiliary function to continue training from last trained models
    def load_last_model(self, checkpoints_path, logging_func=print):
        name = self.model.__class__.__name__
        models = glob(f'{checkpoints_path}/*.pth')
        model_ids = []
        for f in models:
            run_name = Path(f).stem
            # model_name, dataset, _, _, latent_sz, _, epoch = run_name.split('_')
            model_name, dataset, epoch = run_name.split('_')
            print('-------current epoch: ', epoch)
            model_ids.append((int(epoch), f))
        
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
                     checkpoints_path='',
                     logs_path='',
                     images_path='',
                     estimation_dir='',
                     logging_func=print, start_epoch=None):
        
        if self.normalize_data:
            self.calculate_scaling_factor(train_loader)
        
        if reload_model:
            start_epoch = self.load_last_model(checkpoints_path, logging_func)
        else:
            start_epoch = 1
        name = self.model.__class__.__name__
        run_name = f'{'DisentangledVAE'}_{'VCTK'}'.replace(".", "-")}'
        writer = SummaryWriter(f'{logs_path}/{run_name}')
        for epoch in range(start_epoch, start_epoch + epochs):
            print('kl coef: ', self.kl_cof)
            total_recons_loss1, total_recons_loss2, total_recons_loss1_hat, total_recons_loss2_hat,total_z1_kl_loss, total_z2_kl_loss, total_z_style_kl = self.train(train_loader, epoch, logging_func)
            
            print('recons loss1 epoch_{}: {}'.format(epoch, total_recons_loss1 / len(train_loader)))
            print('recons loss2 epoch_{}: {}'.format(epoch, total_recons_loss2 / len(train_loader)))
            print('recons loss1 hat epoch_{}: {}'.format(epoch, total_recons_loss1_hat / len(train_loader)))
            print('recons loss2 hat epoch_{}: {}'.format(epoch, total_recons_loss2_hat / len(train_loader)))
            print('Z1 KL loss epoch_{}: {}'.format(epoch, total_z1_kl_loss / len(train_loader)))
            print('Z2 kL loss epoch_{}: {}'.format(epoch, total_z2_kl_loss / len(train_loader)))
            print('Z Style KL epoch_{}: {}'.format(epoch, total_z_style_kl / len(train_loader)))

            writer.add_scalar('Loss\Reconstruction Loss1', total_recons_loss1 / len(train_loader), epoch)
            writer.add_scalar('Loss\Reconstruction Loss2', total_recons_loss2 / len(train_loader), epoch)
            writer.add_scalar('Loss\Z1 KL Loss', total_z1_kl_loss / len(train_loader), epoch)
            writer.add_scalar('Loss\Z2 KL Loss', total_z2_kl_loss / len(train_loader), epoch)
            writer.add_scalar('Loss\Z KL Style', total_z_style_kl / len(train_loader), epoch)

            # For each report interval store model and save images
            if epoch % report_interval == 0:
                if not os.path.exists(images_path):
                    os.mkdir(images_path)
                if not os.path.exists(checkpoints_path):
                    os.mkdir(checkpoints_path)
                with torch.no_grad():
                    torch.save(self.model.state_dict(), 
                               f'{checkpoints_path}/{run_name}_{epoch}.pth')
                    self.estimate_trained_model(test_loader, checkpoints_path, estimation_dir)


    def estimate_trained_model(self, test_loader, checkpoints_path, estimation_dir):

        logging_epoch = self.load_last_model(checkpoints_path, logging_func=print)
        self.model.eval()
        if not os.path.exists(estimation_dir):
            os.mkdir(estimation_dir)

        with torch.no_grad():
            data1, data2,speaker_ids = next(iter(test_loader))
            data1 = data1.to(torch.device("cuda")).float()
            data2 = data2.to(torch.device("cuda")).float()
            speaker_ids = speaker_ids.view(-1)

            _,_,recons_x1, recons_x2, _, _, _,_,_,_ = \
            self.model(data1, data2,train=False)
            
            for i in range(10):

                original_mel_fp = os.path.join(estimation_dir, str(logging_epoch) + '_original_mel_' +str(i)+'.png')
                recons_mel_fp = os.path.join(estimation_dir, str(logging_epoch) + '_recons_mel_' +str(i)+'.png')
                recons_mel = recons_x1[i]
                origin_mel = data1[i]

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



    def voice_conversion_mel(self, ckp_path, generation_dir, src_spk, trg_spk,dataset_fp=''):

        source_speaker= src_spk
        target_speaker= trg_spk
        save_dir = os.path.join(generation_dir, source_speaker +'_to_'+target_speaker)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        epoch = self.load_last_model(ckp_path, logging_func=print)
        self.model.eval()
        vocoder_model = build_model().to(device)
        ckpt = torch.load('/home/ubuntu/checkpoint_step001000000_ema.pth')
        vocoder_model.load_state_dict(ckpt['state_dict'])
        #################################################################################
        source_utt_fp = glob(os.path.join(dataset_fp, src_spk, "*.npy"))
        source_utt_fp = np.sort(source_utt_fp)
        target_utt_fp = glob(os.path.join(dataset_fp, trg_spk, '*.npy'))
        print('--------------- len: ', len(source_utt_fp))
        # for i in range(10):
        for utt_fp in source_utt_fp:
            
            source_mel = np.load(source_utt_fp[i])
            source_mel = chunking_mel(source_mel).cuda().float()
            rnd_trg = np.random.choice(len(target_utt_fp), 1)[0]
            target_mel = np.load(target_utt_fp[rnd_trg])
            target_mel = chunking_mel(target_mel).cuda().float()

            
            utterance_id = source_utt_fp[i].split('/')[-1].split('.')[0].split("_")[-1]
            print('convert utterance: {} from --->{} to --->{}'.format(utterance_id, src_spk, trg_spk))
            with torch.no_grad():
                src_style_mu, src_style_logvar, src_content_mu, src_content_logvar = self.model.encode(source_mel)
                trg_style_mu, trg_style_logvar, trg_content_mu, trg_content_logvar = self.model.encode(target_mel)

                src_style = torch.mean(src_style_mu, axis=0, keepdim=True).repeat(source_mel.shape[0], 1)
                trg_style = torch.mean(trg_style_mu, axis=0, keepdim=True).repeat(source_mel.shape[0], 1)
                
                source_z = torch.cat([src_style, src_content_mu], dim=-1)
                convert_z = torch.cat([trg_style, src_content_mu], dim=-1)
             
                recons_mel = self.model.decode(source_z)
                recons_voice = torch.cat([recons_mel[i] for i in range(recons_mel.shape[0])], 1)
                recons_voice = recons_voice.cpu().detach().numpy()

                converted_mel = self.model.decode(convert_z)
                converted_mel_hat = self.model.postnet(converted_mel)
                converted_mel = converted_mel + converted_mel_hat

                converted_voice = torch.cat([converted_mel[i] for i in range(converted_mel.shape[0])], 1)
                converted_voice = converted_voice.cpu().detach().numpy()

                source_mel = torch.cat([source_mel[i] for i in range(source_mel.shape[0])], 1)
                source_mel = source_mel.cpu().detach().numpy()

                spectral_detail = np.multiply(source_mel, np.divide(recons_voice, converted_voice))
                plt.figure()
                plt.title('original_' + source_speaker+'_'+ utterance_id)
                librosa.display.specshow(source_mel, x_axis='time', y_axis='mel', sr=16000)
                plt.colorbar(format='%f')
                plt.savefig(os.path.join(save_dir, 'original_' + source_speaker +'_'+ utterance_id+'.png'))

                plt.figure()
                plt.title('convert_' + source_speaker +'_'+ target_speaker+'_'+ utterance_id)
                librosa.display.specshow(converted_voice, x_axis='time', y_axis='mel', sr=16000)
                plt.colorbar(format='%f')
                plt.savefig(os.path.join(save_dir, 'convert_' + source_speaker +'_'+ target_speaker+'_'+ utterance_id+'.png'))

                plt.figure()
                plt.title('reconstruct_' + source_speaker +'_' + utterance_id)
                librosa.display.specshow(recons_voice, x_axis='time', y_axis='mel', sr=16000)
                plt.colorbar(format='%f')
                plt.savefig(os.path.join(save_dir, 'recons_'+ source_speaker +'_' + utterance_id + '.png'))

                converted_voice = np.transpose(converted_voice, (-1, -2))
                spectral_detail = np.transpose(spectral_detail, (-1, -2))
                recons_voice = np.transpose(recons_voice, (-1, -2))
                source_mel = np.transpose(source_mel, (-1, -2))

                waveform = wavegen(vocoder_model, converted_voice)

                librosa.output.write_wav(os.path.join(save_dir,
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

def chunking_mcc(mcc, length=128):
    data = []
    num_mcc = (mcc.shape[1]//length)+1
    for index in range(num_mcc):
        if index < num_mcc - 1:
            mcc_partition = mcc[:,index*length:index*length+length]
        else:
            mcc_partition = mcc[:,index*length:]
            mcc_partition = np.pad(mcc_partition, ((0,0),(0, length - mcc.shape[1]%length)), 'constant', constant_values=(0,0))
        data.append(mcc_partition)
    return torch.tensor(data)

