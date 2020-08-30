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
from sparse_encoding.train_feature_selection import feature_selection
from sparse_encoding.feature_selection import FeatureSelection
from sparse_encoding import utils
from sparse_encoding import conv_mulvae
from tqdm import tqdm


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
    
    
    def step(self, data, speaker_ids, train=False):
        if train:
            self.optimizer.zero_grad()
        
        # data = data.view(shape[0]*shape[1], shape[2], shape[3])
        output, content_mu, content_logvar, group_style_mu, group_style_logvar =\
        self.model(data, speaker_ids)
        output =  torch.clamp(output, min=0, max=1.0)

        loss, recons_loss, style_kl_loss, content_kl_loss = \
        self.loss_function(data, output,content_mu,content_logvar,group_style_mu, group_style_logvar,train=train)
        if train:
            loss.backward()
            self.optimizer.step()
        return loss.item(), recons_loss.item(), style_kl_loss.item(), content_kl_loss.item()
    
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
        total_recons_loss, total_style_loss, total_content_loss = 0, 0, 0
        for batch_idx, (data, _,speaker_ids) in enumerate(tqdm(train_loader)):
            
            data = data.to(torch.device("cuda")).float()
            data = data.view(-1, data.shape[-2], data.shape[-1])
            # print('speaker_ids: ', speaker_ids)
            # print('dataset shape: ', data.shape)
            speaker_ids = speaker_ids.view(-1)


            loss, recons_loss, style_kl_loss, content_kl_loss = self.step(data, speaker_ids,train=True)
            train_loss += loss
            total_style_loss += style_kl_loss
            total_recons_loss += recons_loss
            total_content_loss += content_kl_loss

            if (batch_idx+1) % self.log_interval == 0:
                logging_func('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}' \
                      .format(epoch, batch_idx * len(data), 
                              len(train_loader.dataset),
                              100. * batch_idx / len(train_loader),
                              loss / len(data)))

        logging_func('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / len(train_loader.dataset)))
        
        return total_recons_loss, total_style_loss, total_content_loss
        
        
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
            total_recons_loss, total_style_loss, total_content_loss = self.train(train_loader, epoch, logging_func)
            # test_loss = self.test(test_loader, epoch, logging_func)
            total_recons_loss /= len(train_loader)
            # total_prior_loss /= len(train_loader)
            # logger.scalar_summary(train_loss, test_loss, epoch)
            # logger.write_log(total_recons_loss, total_prior_loss, epoch)
            print('recons loss epoch_{}: {}'.format(epoch, total_recons_loss / len(train_loader)))
            print('style loss epoch_{}: {}'.format(epoch, total_style_loss / len(train_loader)))
            print('content loss epoch_{}: {}'.format(epoch, total_content_loss / len(train_loader)))
            writer.add_scalar('Loss\Reconstruction Loss', total_recons_loss / len(train_loader), epoch)
            writer.add_scalar('Loss\Style Loss', total_style_loss / len(train_loader), epoch)
            writer.add_scalar('Loss\Content Loss', total_content_loss / len(train_loader), epoch)
            # Optional update
            # self.update_()
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
        # estimate_dir = '../results2/images/estimation'
        if not os.path.exists(estimation_dir):
            os.mkdir(estimation_dir)

        with torch.no_grad():
            data,_,speaker_ids = next(iter(test_loader))
            shape = data.shape
            # data = data.view(shape[0]*shape[1], shape[2], shape[3])
            data = data.to(torch.device("cuda")).float()
            data = data.view(-1, data.shape[-2], data.shape[-1])
            # print('speaker_ids: ', speaker_ids)
            speaker_ids = speaker_ids.view(-1)

            recons_data, content_mu, content_logvar, group_style_mu, group_style_logvar = \
            self.model(data, speaker_ids,train=False)
            
            for i in range(5):

                original_mel_fp = os.path.join(estimation_dir, str(logging_epoch) + '_original_mel_' +str(i)+'.png')
                recons_mel_fp = os.path.join(estimation_dir, str(logging_epoch) + '_recons_mel_' +str(i)+'.png')
                recons_mel = recons_data[i]
                origin_mel = data[i]
                # z_sample = self.model.reparameterize(mu[i], logvar[i], logspike[i])
                # z = mu[i]

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

                # encoding_visualization(z, estimation_dir, i, logging_epoch, prefix='')
                # encoding_visualization(z_sample, estimation_dir, i, logging_epoch, prefix='z_sample')


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

        if not os.path.exists(estimation_dir):
            os.mkdir(estimation_dir)

        epoch = self.load_last_model(ckp_path, logging_func=print)
        self.model.eval()
        if utterance == None:
            batch_data = dataset.get_batch_utterances(speaker_id, 100)
            batch_data =  batch_data.cuda().float()
        # else:
        #     batch_data,_,_ = dataset.get_batch_speaker(utterance)
        #     batch_data =  batch_data.cuda().float()
        #     speaker_id = 'analysi_utt' + utterance.split('.')[0]
        else:
            mel_spec = dataset.get_utterance(speaker_id,utterance)
            batch_data = chunking_mel(mel_spec).cuda().float()

        _,_, mu, logvar, logspike = self.model(batch_data)
        latent_vectors = self.model.reparameterize(mu, logvar, logspike)
        batch_idx = []
        for idx in range(latent_vectors.shape[0]):
            index = encoding_visualization(latent_vectors[idx,:], estimation_dir, idx, epoch,'')
            batch_idx.append(index)
        
        print('mutal index: ', np.intersect1d(batch_idx[3], batch_idx[4],
                                             batch_idx[5], batch_idx[6]))

        # plot_latentvt_analysis(latent_vectors, estimation_dir, speaker_id, threshold_mean=0.1, threshold_std=0.7)

    def voice_conversion(self, target_speaker, source_speaker,
                        utterance_id, dataset, ckp_path, generation_dir):

        epoch = self.load_last_model(ckp_path, logging_func=print)
        self.model.eval()
        vocoder_model = build_model().to(device)
        ckpt = torch.load('/home/ubuntu/checkpoint_step001000000_ema.pth')
        vocoder_model.load_state_dict(ckpt['state_dict'])

        source_data, source_spk_id = dataset.get_batch_utterances(source_speaker, 1)
        source_data = source_data.cuda().float()
        target_data, target_spk_id = dataset.get_batch_utterances(target_speaker, 1)
        target_data = target_data.cuda().float()
        utterance = dataset.get_utterance(source_speaker, utterance_id)
        utterance = chunking_mel(utterance).cuda().float()

        with torch.no_grad():
            source_style_mu, source_style_logvar,_,_ = self.model.encode(source_data)
            target_style_mu, target_style_logvar,_,_ = self.model.encode(target_data)
            
            group_source_style_mu, group_source_style_logvar = utils.accumulate_group_evidence(source_style_mu, source_style_logvar, source_spk_id, True)
            group_target_style_mu, group_target_style_logvar = utils.accumulate_group_evidence(target_style_mu, target_style_logvar, target_spk_id, True)
            group_source_style_mu = torch.unique(group_source_style_mu, dim=0)
            group_target_style_mu = torch.unique(group_target_style_mu, dim=0)
            group_target_style_logvar = torch.unique(group_target_style_logvar, dim=0)
            group_source_style_logvar = torch.unique(group_source_style_logvar, dim=0)
            

            group_source_style_z = self.model._reparameterize(group_source_style_mu, group_source_style_logvar)
            group_target_style_z = self.model._reparameterize(group_target_style_mu, group_target_style_logvar)
            # group_source_style_z = group_source_style_mu
            # group_target_style_z = group_target_style_mu
            print('source style mu: ', group_source_style_z)
            print('target style mu: ', group_target_style_z)
            group_target_style_z = conv_mulvae.tile(group_target_style_z, 0, utterance.shape[0])
            group_source_style_z = conv_mulvae.tile(group_source_style_z, 0, utterance.shape[0])

            style_mu, style_logvar, content_mu, content_logvar = self.model.encode(utterance)
            print('content mu: ', content_mu[0])
            recons_mel = self.model.decode(content_mu, group_source_style_z)
            converted_mel = self.model.decode(content_mu, group_target_style_z)    
            
            # recons_mel = self.model.decode(b)
            recons_voice = torch.cat([recons_mel[i] for i in range(recons_mel.shape[0])], 1)
            recons_voice = recons_voice.cpu().detach().numpy()
            # converted_voice = np.transpose(converted_voice, (-1, -2))

            # converted_mel = self.model.decode(mu)
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

            # recons_voice = np.transpose(recons_voice, (-1, -2))
            # recons_waveform = wavegen(vocoder_model, recons_voice)

            librosa.output.write_wav(os.path.join(generation_dir,
            'convert_'+source_speaker+'_to_'+target_speaker+'_'+utterance_id.split('.')[0]+'.wav'), waveform, sr=16000)
            # librosa.output.write_wav(os.path.join(generation_dir,
            # 'recons_'+source_speaker+'_'+utterance_id.split('.')[0]+'.wav'), recons_waveform, sr=16000)

    def voice_conversion2(self, target_speaker, source_speaker,
                        utterance_id, dataset, ckp_path, generation_dir):

        epoch = self.load_last_model(ckp_path, logging_func=print)
        self.model.eval()
        vocoder_model = build_model().to(device)
        ckpt = torch.load('/home/ubuntu/checkpoint_step001000000_ema.pth')
        vocoder_model.load_state_dict(ckpt['state_dict'])
        source_data = dataset.get_batch_utterances(source_speaker, 10).cuda().float()
        target_data = dataset.get_batch_utterances(target_speaker, 10).cuda().float()
        utterance = dataset.get_utterance(source_speaker, utterance_id)
        utterance = chunking_mel(utterance).cuda().float()

        with torch.no_grad():
            _,_, source_mu, source_logvar, source_logspike = self.model(source_data)
            _,_, target_mu, target_logvar, target_logspike = self.model(target_data)

            source_z = self.model.reparameterize(source_mu, source_logvar, source_logspike)
            target_z = self.model.reparameterize(target_mu, target_logvar, target_logspike)
            
            source_idx, source_mean, source_std = plot_latentvt_analysis(source_z,
                                                                            generation_dir, source_speaker, 0.1, 0.6)
            target_idx, target_mean, target_std = plot_latentvt_analysis(target_z,
                                                                            generation_dir, target_speaker, 0.1, 0.6)
            target_mean = torch.from_numpy(target_mean)
            source_mean = torch.from_numpy(source_mean)
            mu, logvar, logspike = self.model.encode(utterance)

            mutual_idx = np.intersect1d(source_idx, target_idx)

            z = self.model.reparameterize(mu, logvar, logspike)
            backup_z = z.clone()
            print('target mean: ', target_mean)
            print('target index: ', target_idx)
            print('source index: ', source_idx)
            print('mutual index: ', mutual_idx)

            for j in range(mu.shape[0]):
                for i, idx in enumerate(source_idx):
                    z[j, idx] = 0

            for j in range(mu.shape[0]):    
                for i, idx in enumerate(target_idx):
                    z[j,idx] = target_mean[i]

            recons_mel = self.model.decode(backup_z)
            recons_voice = torch.cat([recons_mel[i] for i in range(recons_mel.shape[0])], 1)
            recons_voice = recons_voice.cpu().detach().numpy()
            # converted_voice = np.transpose(converted_voice, (-1, -2))

            converted_mel = self.model.decode(z)
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

            # recons_voice = np.transpose(recons_voice, (-1, -2))
            # recons_waveform = wavegen(vocoder_model, recons_voice)

            librosa.output.write_wav(os.path.join(generation_dir,
            'convert_'+source_speaker+'_to_'+target_speaker+'_'+utterance_id.split('.')[0]+'.wav'), waveform, sr=16000)
            # librosa.output.write_wav(os.path.join(generation_dir,
            # 'recons_'+source_speaker+'_'+utterance_id.split('.')[0]+'.wav'), recons_waveform, sr=16000)

    def voice_conversion3(self, target_utterance, source_utterance,
                          dataset, ckp_path, generation_dir):
        target_speaker = 'vcc2016_training_SM1'
        epoch = self.load_last_model(ckp_path, logging_func=print)
        self.model.eval()
        vocoder_model = build_model().to(device)
        ckpt = torch.load('/home/ubuntu/checkpoint_step001000000_ema.pth')
        vocoder_model.load_state_dict(ckpt['state_dict'])
        ########################### load feature selection model #######################
        ckp_fp = '/vinai/manhlt/icassp-20/icassp-20/VC_logs2/VCTK_Autovc_512_64/fs_ckp/'
        list_ckp = glob(os.path.join(ckp_fp, "*.pth"))
        ckp_fp = list_ckp[-1]
        fs_model = FeatureSelection(512, 109).cuda()
        fs_model.load_state_dict(torch.load(ckp_fp))
        fs_model.eval()
        #################################################################################
        source_mel = np.load(source_utterance)
        source_mel = chunking_mel(source_mel).cuda().float()
        # target_mel = dataset.get_batch_utterances(target_speaker, 10).cuda().float()
        target_mel = np.load(target_utterance)
        target_mel = chunking_mel(target_mel).cuda().float()

        source_speaker= 'SF1'
        target_speaker= 'SM1'
        utterance_id = '100002'

        with torch.no_grad():
            _,_, source_mu, source_logvar, source_logspike = self.model(source_mel)
            _,_, target_mu, target_logvar, target_logspike = self.model(target_mel)

            source_z = self.model.reparameterize(source_mu, source_logvar, source_logspike)
            target_z = self.model.reparameterize(target_mu, target_logvar, target_logspike)
            
            mean_target_z = torch.mean(target_z, dim=0)

            source_idx = feature_selection(fs_model, source_z)
            target_idx = feature_selection(fs_model, target_z)
            print('source_idx: ', source_idx.shape)
            print('target idx: ', target_idx.shape)
            # print('mean target z: ', mean_target_z > 0.5)
            
            backup_z = source_mu.clone()

            for j in range(source_z.shape[0]):
                for i, idx in enumerate(source_idx):
                    source_z[j, idx] = 0
            
            for j in range(source_z.shape[0]):
                for i, idx in enumerate(target_idx):
                    source_z[j, idx] = mean_target_z[idx]

            
            recons_mel = self.model.decode(backup_z)
            recons_voice = torch.cat([recons_mel[i] for i in range(recons_mel.shape[0])], 1)
            recons_voice = recons_voice.cpu().detach().numpy()

            converted_mel = self.model.decode(source_z)
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

            # recons_voice = np.transpose(recons_voice, (-1, -2))
            # recons_waveform = wavegen(vocoder_model, recons_voice)

            librosa.output.write_wav(os.path.join(generation_dir,
            'convert_'+source_speaker+'_to_'+target_speaker+'_'+utterance_id.split('.')[0]+'.wav'), waveform, sr=16000)
            # librosa.output.write_wav(os.path.join(generation_dir,
            # 'recons_'+source_speaker+'_'+utterance_id.split('.')[0]+'.wav'), recons_waveform, sr=16000)


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

