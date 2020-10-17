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
from sparse_encoding.train_feature_selection import feature_selection
from sparse_encoding.feature_selection import FeatureSelection
from sparse_encoding import utils
from sparse_encoding import conv_mulvae_mel
from tqdm import tqdm
from tensorboardX import SummaryWriter
from preprocessing.processing import build_model, wavegen
from preprocessing.WORLD_processing import *
import numpy as np
import librosa




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VariationalBaseModelGVAE():
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
        
        # recons_x1, recons_x2, q_z1_mu, q_z1_logvar, q_z2_mu, q_z2_logvar, style_mu1, style_logvar1 =\
        # self.model(data1, data2)

        recons_x1, recons_x2, recons_x1_hat,recons_x2_hat,q_z1_mu, q_z1_logvar, q_z2_mu, q_z2_logvar, style_mu, style_logvar =\
        self.model(data1, data2)

        # loss, recons_loss1, recons_loss2, z1_kl_loss, z2_kl_loss, z_style_kl = \
        # self.loss_functionGVAE2(data1, data2, recons_x1, recons_x2,q_z1_mu,q_z1_logvar,q_z2_mu, q_z2_logvar, style_mu1, style_logvar1,train=train)

        loss, recons_loss1, recons_loss2, recons_loss1_hat, recons_loss2_hat, z1_kl_loss, z2_kl_loss, z_style_kl = \
        self.loss_functionGVAE2(data1,data2,recons_x1, recons_x2, recons_x1_hat, recons_x2_hat,q_z1_mu,q_z1_logvar,q_z2_mu, q_z2_logvar,style_mu, style_logvar,train=train)
        if train:
            loss.backward()
            self.optimizer.step()
        return loss.item(), recons_loss1.item(), recons_loss2.item(), recons_loss1_hat.item(), recons_loss2_hat.item(), z1_kl_loss.item(), z2_kl_loss.item(), z_style_kl.item()
    
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
        train_loss, total_z_style_kl = 0, 0
        total_recons_loss1, total_recons_loss2, total_z1_kl_loss, total_z2_kl_loss = 0, 0, 0, 0
        total_recons_loss1_hat, total_recons_loss2_hat = 0,0
        for batch_idx, (data1, data2, speaker_ids) in enumerate(tqdm(train_loader)):
            
            data1 = data1.to(torch.device("cuda")).float()
            # data1 = data1.view(-1, data1.shape[-2], data1.shape[-1])

            data2 = data2.to(torch.device("cuda")).float()
            # data2 = data2.view(-1, data2.shape[-2], data2.shape[-1])
            # print('speaker_ids: ', speaker_ids)
            # print('---------------dataset shape: ', data1.shape)
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

            # if (batch_idx+1) % self.log_interval == 0:
            #     logging_func('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}' \
            #           .format(epoch, batch_idx * len(data), 
            #                   len(train_loader.dataset),
            #                   100. * batch_idx / len(train_loader),
            #                   loss / len(data)))

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
            model_name, dataset, _, _, latent_sz, _, epoch = run_name.split('_')
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

    def load_model(self, checkpoints_path, epoch,logging_func=print):
        name = self.model.__class__.__name__
        model_name = 'MulVAE_mnist_1701_200000000000000000_32_0-0001_' + str(epoch) +'.pth'
        last_checkpoint = os.path.join(checkpoints_path, model_name)
        # logging_func('Last checkpoint: ', last_checkpoint)
        self.model.load_state_dict(torch.load(last_checkpoint))
        logging_func(f'Loading {name} model from last checkpoint ({epoch})...')

        return epoch
    
    
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
        # logger = Logger(f'{logs_path}/{run_name}')f
        # logging_func(f'Training {name} model...')
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
            # Optional update
            # self.update_()
            # For each report interval store model and save images
            if epoch % report_interval == 0:
                # self.update_kl()
                if not os.path.exists(images_path):
                    os.mkdir(images_path)
                if not os.path.exists(checkpoints_path):
                    os.mkdir(checkpoints_path)
                with torch.no_grad():
                    torch.save(self.model.state_dict(), 
                               f'{checkpoints_path}/{run_name}_{epoch}.pth')
                    self.estimate_trained_model(test_loader, checkpoints_path, estimation_dir)

            # if epoch == 120:
            #     self.set_kl(1)
            # if epoch % 50 == 0:
            #     self.update_kl()

    def estimate_trained_model(self, test_loader, checkpoints_path, estimation_dir):

        logging_epoch = self.load_last_model(checkpoints_path, logging_func=print)
        self.model.eval()
        # estimate_dir = '../results2/images/estimation'
        if not os.path.exists(estimation_dir):
            os.mkdir(estimation_dir)

        with torch.no_grad():
            data1, data2,speaker_ids = next(iter(test_loader))
            # shape = data.shape
            # data = data.view(shape[0]*shape[1], shape[2], shape[3])
            data1 = data1.to(torch.device("cuda")).float()
            # data1 = data1.view(-1, data1.shape[-2], data1.shape[-1])
            data2 = data2.to(torch.device("cuda")).float()
            # data2 = data2.view(-1, data.shape[-2], data.shape[-1])

            speaker_ids = speaker_ids.view(-1)

            _,_,recons_x1, recons_x2, _, _, _,_,_,_ = \
            self.model(data1, data2,train=False)
            
            for i in range(2):

                original_mel_fp = os.path.join(estimation_dir, str(logging_epoch) + '_original_mel_' +str(i)+'.png')
                recons_mel_fp = os.path.join(estimation_dir, str(logging_epoch) + '_recons_mel_' +str(i)+'.png')
                recons_mel = recons_x1[i]
                origin_mel = data1[i]
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

    def voice_conversion_mcc(self, source_spk, target_spk, source_utt, target_utt,
                            dataset, wav_fp, ckp_path, generation_dir):

        epoch = self.load_last_model(ckp_path, logging_func=print)
        self.model.eval()
        utterance_id = wav_fp.split('/')[-1].split('.')[0]

        source = dataset.get_utterance(source_spk, source_utt)
        print('source input shape: ', source['normalized_mc'].shape)
        source_mcc_norm = chunking_mcc(source['normalized_mc'].T, 128).cuda().float()
        print('source_mcc_norm shape: ', source_mcc_norm.shape)
        source_mcc_mean, source_mcc_std = source['mc_mean'], source['mc_std']
        source_f0s = source['f0']
        source_f0_mean, source_f0_std = logf0_statistics(source_f0s)

        target = dataset.get_utterance(target_spk, target_utt)
        target_mcc_norm = chunking_mcc(target['normalized_mc'].T, 128).cuda().float()
        target_f0s = target['f0']
        target_f0_mean, target_f0_std = logf0_statistics(target_f0s)

        f0_converted = pitch_conversion(f0=source_f0s, mean_log_src=source_f0_mean, std_log_src=source_f0_std,
                                        mean_log_target=target_f0_mean, std_log_target=target_f0_std)
        wav, _ = librosa.load(wav_fp, sr=16000)
        f0, timeaxis, sp, ap, mc = world_encode_data(wav=wav, fs=16000)

        with torch.no_grad():
            source_style_mu, source_style_logvar, source_content_mu, source_content_logvar = self.model.encode(source_mcc_norm)
            target_style_mu, target_style_logvar, target_content_mu, target_content_logvar = self.model.encode(target_mcc_norm)
            
            source_style = torch.mean(source_style_mu, axis=0, keepdim=True).repeat(source_mcc_norm.shape[0], 1)
            target_style = torch.mean(target_style_mu, axis=0, keepdim=True).repeat(source_mcc_norm.shape[0], 1)

            source_z = torch.cat([source_style, source_content_mu], dim=-1)
            convert_z = torch.cat([target_style, source_content_mu], dim=-1)

            
            recons_mcc = self.model.decode(source_z)
            converted_mcc = self.model.decode(convert_z)    
            
            recons_voice = torch.cat([recons_mcc[i] for i in range(recons_mcc.shape[0])], 1)
            recons_voice = recons_voice.cpu().detach().numpy()

            converted_voice = torch.cat([converted_mcc[i] for i in range(converted_mcc.shape[0])], 1)
            converted_voice = converted_voice.cpu().detach().numpy()

            plt.figure()
            plt.title('convert_' + source_spk +'_'+ target_spk+'_'+ utterance_id)
            librosa.display.specshow(converted_voice, x_axis='time', y_axis='mel', sr=16000)
            plt.colorbar(format='%f')
            plt.savefig(os.path.join(generation_dir, 'convert_' + source_spk +'_'+ target_spk+'_'+ utterance_id+'.png'))

            plt.figure()
            plt.title('reconstruct_' + source_spk +'_' + utterance_id)
            librosa.display.specshow(recons_voice, x_axis='time', y_axis='mel', sr=16000)
            plt.colorbar(format='%f')
            plt.savefig(os.path.join(generation_dir, 'recons_'+ source_spk +'_' + utterance_id + '.png'))
            
            recons_voice_mcc = recons_voice.T[:664,:] *source_mcc_std + source_mcc_mean
            converted_voice_mcc = converted_voice.T[:664,:]*source_mcc_std + source_mcc_mean
            # recons_voice_mcc = recons_voice.T[:808,:].astype(np.float)
            # converted_voice_mcc = converted_voice.T[:808,:].astype(np.float)

            recons_sp = world_decode_mc(np.ascontiguousarray(recons_voice_mcc), 16000)
            converted_sp = world_decode_mc(np.ascontiguousarray(converted_voice_mcc), 16000)
            
            recons_wav = world_speech_synthesis(f0=f0, sp=recons_sp, ap=ap, fs=16000, frame_period=5.0)
            converted_wav = world_speech_synthesis(f0=f0_converted, sp=converted_sp, ap=ap, fs=16000, frame_period=5.0)


            librosa.output.write_wav(os.path.join(generation_dir,
            'convert_'+source_spk+'_to_'+target_spk+'_'+utterance_id+'.wav'), converted_wav, sr=16000)

            librosa.output.write_wav(os.path.join(generation_dir,
            'recons_'+source_spk+'_'+ utterance_id +'.wav'), recons_wav, sr=16000)

    def vc_evaluation(self, source_speaker, target_speaker,
                     evaluation_fp, ckp_path, dataset_fp, SR=16000):

        if not os.path.exists(os.path.join(evaluation_fp, 'wav',source_speaker + '_to_'+target_speaker)):
            os.mkdir(os.path.join(evaluation_fp, 'wav',source_speaker + '_to_'+target_speaker))
        fp1 = os.path.join(evaluation_fp, 'wav',source_speaker + '_to_'+target_speaker)

        if not os.path.exists(os.path.join(evaluation_fp, 'mcep',source_speaker + '_to_'+target_speaker)):
            os.mkdir(os.path.join(evaluation_fp, 'mcep',source_speaker + '_to_'+target_speaker))
        fp2 = os.path.join(evaluation_fp, 'mcep',source_speaker + '_to_'+target_speaker)

        epoch = self.load_last_model(ckp_path, logging_func=print)
        # epoch = self.load_model(ckp_path, 9900, logging_func=print)
        self.model.eval()

        source_utt_fp = glob(os.path.join(dataset_fp, source_speaker, '*.npz'))
        rnd_idx = np.random.shuffle(source_utt_fp)
        target_utt_fp = glob(os.path.join(dataset_fp, target_speaker, '*.npz'))
        # source_utt_fp = sorted(source_utt_fp)
        # target_utt_fp = sorted(target_utt_fp)

        stat_data_fp = os.path.join('/home/ubuntu/vcc2018_stat_data/')  

        src_logf0 = np.load(stat_data_fp + 'log_f0_' + source_speaker + '.npz')
        # src_logf0_mean = src_logf0['mean']
        # src_logf0_std = src_logf0['std']
        src_mcep = np.load(stat_data_fp + 'mcep_' + source_speaker + '.npz')
        src_mcep_mean = src_mcep['mean']
        scr_mcep_std = src_mcep['std']

        ''' ------------------for known speakers------------------------------------ '''
        trg_logf0 = np.load(stat_data_fp + 'log_f0_' + target_speaker + '.npz')
        # trg_logf0_mean = trg_logf0['mean']
        # trg_logf0_std = trg_logf0['std']
        trg_mcep = np.load(stat_data_fp + 'mcep_' + target_speaker + '.npz')
        trg_mcep_mean = trg_mcep['mean']
        trg_mcep_std = trg_mcep['std']
        ''' -----------------for unknown speakers------------------------------------'''
        # trg_utt_fp = os.path.join('/home/ubuntu/vcc2016_training/', target_speaker)

        for utt_fp in source_utt_fp:
            
            rnd_trg = np.random.choice(len(target_utt_fp), 1)[0]
            utt1 = np.load(utt_fp)
            src_mcep_mean = utt1['mc_mean']
            src_mcep_std = utt1['mc_std']
            utt2 = np.load(target_utt_fp[rnd_trg])
            trg_mcc = utt2['normalized_mc']
            trg_ap = utt2['ap']

            src_mcc = utt1['normalized_mc']
            print('source mcc shape: ', src_mcc.shape)
            # print('length of target files: ', len(target_utt_fp))
            # print('random index: ', rnd_trg)
            voice_length = src_mcc.shape[0]
            src_ap = utt1['ap']
            src_f0 = utt1['f0']
            src_sp = utt1['sp']
            utt_id = utt_fp.split('/')[-1].split('.')[0]
            '''------------------------extract logF0 of target utterances--------------------------------------'''
            # trg_utt_id = target_utt_fp[rnd_trg].split('/')[-1].split('.')[0]
            # trg_wav,_ = librosa.load(os.path.join(trg_utt_fp, trg_utt_id +'.wav'), sr=16000)
            # f0, timeaxis, sp, ap, mc = world_encode_data(wav=trg_wav.astype(np.float), fs=16000)
            # log_f0 = np.ma.log(f0)
            trg_logf0 = np.ma.log(utt2['f0'])
            trg_logf0_mean = trg_logf0.mean()
            trg_logf0_std = trg_logf0.std()
            src_logf0 = np.ma.log(src_f0)
            src_logf0_mean = src_logf0.mean()
            src_logf0_std = src_logf0.std()
            print(src_logf0_mean)
            print(trg_logf0_mean)
            '''------------------------------------------------------------------------------------------------'''
            

            src_mcc_chunk = chunking_mcc(src_mcc.T, 128).cuda().float()
            trg_mcc_chunk = chunking_mcc(trg_mcc.T, 128).cuda().float()
            src_style_mu, src_style_logvar, src_content_mu, src_content_logvar = self.model.encode(src_mcc_chunk)
            trg_style_mu, trg_style_logvar, trg_content_mu, trg_content_logvar = self.model.encode(trg_mcc_chunk)

            src_style = torch.mean(src_style_mu, axis=0, keepdim=True).repeat(src_mcc_chunk.shape[0], 1)
            trg_style = torch.mean(trg_style_mu, axis=0, keepdim=True).repeat(src_mcc_chunk.shape[0], 1)

            source_z = torch.cat([src_style, src_content_mu], dim=-1)
            convert_z = torch.cat([trg_style, src_content_mu], dim=-1)

            cvt_mcc = self.model.decode(convert_z)
            cvt_voice_mcc = torch.cat([cvt_mcc[i] for i in range(cvt_mcc.shape[0])], 1)
            cvt_voice_mcc = cvt_voice_mcc.cpu().detach().numpy()
            cvt_voice_mcc = cvt_voice_mcc.T.astype(np.double)
            # cvt_voice_mcc = cvt_voice_mcc[:voice_length,:]    

            '''origin code'''
            # cvt_voice_mcc = cvt_voice_mcc[:voice_length,:]*src_mcep_std + src_mcep_mean
            # cvt_sp = world_decode_mc(np.ascontiguousarray(cvt_voice_mcc), SR)
            # converted_f0 = pitch_conversion(f0=src_f0, mean_log_src=src_logf0_mean, std_log_src=src_logf0_std,mean_log_target=trg_logf0_mean, std_log_target=trg_logf0_std)
            # cvt_wav = world_speech_synthesis(f0=converted_f0, sp=cvt_sp, ap=src_ap, frame_period=5, fs=SR)
            # wav_fp = os.path.join(fp1, 'cvt_'+str(utt_id)+'.wav')
            # mcep_fp = os.path.join(fp2, str(utt_id) + '.npz')

            # librosa.output.write_wav(wav_fp, cvt_wav, sr=SR)
            # np.savez(mcep_fp, mcc=cvt_voice_mcc, f0=converted_f0)
            ''''''''''''''''''''''''''''''''''''''''''
            cvt_mcc_s = self.model.decode(source_z)
            cvt_voice_mcc_s = torch.cat([cvt_mcc_s[i] for i in range(cvt_mcc_s.shape[0])], 1)
            cvt_voice_mcc_s = cvt_voice_mcc_s.cpu().detach().numpy()
            cvt_voice_mcc_s = cvt_voice_mcc_s.T.astype(np.double)

            cvt_voice_mcc = cvt_voice_mcc[:voice_length,:]*src_mcep_std + src_mcep_mean
            cvt_sp_t = world_decode_mc(np.ascontiguousarray(cvt_voice_mcc), SR)

            cvt_voice_mcc_s = cvt_voice_mcc_s[:voice_length,:]*src_mcep_std + src_mcep_mean
            cvt_sp_s = world_decode_mc(np.ascontiguousarray(cvt_voice_mcc_s), SR)
            
            gained_sp = np.multiply(src_sp, np.divide(cvt_sp_t, cvt_sp_s))
            # print(np.divide(cvt_sp_t, cvt_sp_s))
            converted_f0 = pitch_conversion(f0=src_f0, mean_log_src=src_logf0_mean, std_log_src=src_logf0_std,mean_log_target=trg_logf0_mean, std_log_target=trg_logf0_std)
            cvt_wav = world_speech_synthesis(f0=converted_f0, sp=gained_sp, ap=src_ap, frame_period=5, fs=SR)
            wav_fp = os.path.join(fp1, 'cvt_'+str(utt_id)+'.wav')
            mcep_fp = os.path.join(fp2, str(utt_id) + '.npz')

            librosa.output.write_wav(wav_fp, cvt_wav, sr=SR)
            np.savez(mcep_fp, mcc=cvt_voice_mcc, f0=converted_f0)

    def voice_conversion_mel(self,
                           ckp_path, generation_dir, dataset_fp='/home/ubuntu/VCTK_mel'):
        src_spk = 'VCTK-Corpus_wav16_p225'
        trg_spk = 'VCTK-Corpus_wav16_p226'
        source_speaker= src_spk.split('_')[-1]
        target_speaker= trg_spk.split('_')[-1]
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
        # rnd_idx = np.random.shuffle(source_utt_fp)
        target_utt_fp = glob(os.path.join(dataset_fp, trg_spk, '*.npy'))
        print('--------------- len: ', len(source_utt_fp))
        for i in range(10):
        # for utt_fp in source_utt_fp:
            
            source_mel = np.load(source_utt_fp[i])
            source_mel = chunking_mel(source_mel).cuda().float()
            rnd_trg = np.random.choice(len(target_utt_fp), 1)[0]
            # target_mel = dataset.get_batch_utterances(target_speaker, 10).cuda().float()
            # print('------------------------',rnd_trg)
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
                # recons_mel_hat = self.model.postnet(recons_mel)
                # recons_mel = recons_mel_hat + recons_mel_hat

                recons_voice = torch.cat([recons_mel[i] for i in range(recons_mel.shape[0])], 1)
                recons_voice = recons_voice.cpu().detach().numpy()

                converted_mel = self.model.decode(convert_z)
                converted_mel_hat = self.model.postnet(converted_mel)
                converted_mel = converted_mel + converted_mel_hat


                converted_voice = torch.cat([converted_mel[i] for i in range(converted_mel.shape[0])], 1)
                converted_voice = converted_voice.cpu().detach().numpy()

                source_mel = torch.cat([source_mel[i] for i in range(source_mel.shape[0])], 1)
                source_mel = source_mel.cpu().detach().numpy()

                # print('ratio between converted voice and recons voice: ', np.divide(converted_voice, recons_voice))
                spectral_detail = np.multiply(source_mel, np.divide(recons_voice, converted_voice))
                # plt.figure()
                # plt.title('original_' + source_speaker+'_'+ utterance_id)
                # librosa.display.specshow(source_mel, x_axis='time', y_axis='mel', sr=16000)
                # plt.colorbar(format='%f')
                # plt.savefig(os.path.join(save_dir, 'original_' + source_speaker +'_'+ utterance_id+'.png'))

                # plt.figure()
                # plt.title('convert_' + source_speaker +'_'+ target_speaker+'_'+ utterance_id)
                # librosa.display.specshow(converted_voice, x_axis='time', y_axis='mel', sr=16000)
                # plt.colorbar(format='%f')
                # plt.savefig(os.path.join(save_dir, 'convert_' + source_speaker +'_'+ target_speaker+'_'+ utterance_id+'.png'))

                # plt.figure()
                # plt.title('reconstruct_' + source_speaker +'_' + utterance_id)
                # librosa.display.specshow(recons_voice, x_axis='time', y_axis='mel', sr=16000)
                # plt.colorbar(format='%f')
                # plt.savefig(os.path.join(save_dir, 'recons_'+ source_speaker +'_' + utterance_id + '.png'))

                converted_voice = np.transpose(converted_voice, (-1, -2))
                spectral_detail = np.transpose(spectral_detail, (-1, -2))
                recons_voice = np.transpose(recons_voice, (-1, -2))
                source_mel = np.transpose(source_mel, (-1, -2))

                waveform = wavegen(vocoder_model, converted_voice)
                # recons_waveform = wavegen(vocoder_model, source_mel)

                # recons_voice = np.transpose(recons_voice, (-1, -2))
                # recons_waveform = wavegen(vocoder_model, recons_voice)

                librosa.output.write_wav(os.path.join(save_dir,
                'convert_'+source_speaker+'_to_'+target_speaker+'_'+utterance_id.split('.')[0]+'.wav'), waveform, sr=16000)
                # librosa.output.write_wav(os.path.join(generation_dir,
                # 'recons_'+source_speaker+'_'+utterance_id+'.wav'), recons_waveform, sr=16000)


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
    # print('number mel-cestra coff: ', num_mcc)
    for index in range(num_mcc):
        if index < num_mcc - 1:
            mcc_partition = mcc[:,index*length:index*length+length]
            # print('mcc shape: ', mcc_partition.shape)
        else:
            mcc_partition = mcc[:,index*length:]
            mcc_partition = np.pad(mcc_partition, ((0,0),(0, length - mcc.shape[1]%length)), 'constant', constant_values=(0,0))
            # print('last mcc shape: ', mcc_partition.shape)
        # print('chunking mcc data: ', mcc_partition.shape)
        data.append(mcc_partition)
    return torch.tensor(data)

