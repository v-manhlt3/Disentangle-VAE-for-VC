import torch
from torch.utils.data import DataLoader, Dataset
from preprocessing.encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from pathlib import Path
from preprocessing.dataset_mel import SpeechDataset3
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from autovc_replicate.original_autovc import Encoder, Decoder, Generator
import argparse
import json
from autovc_replicate.speaker_emb import SpeakerEncoder
from pathlib import Path
import math
import numpy as np
from preprocessing.processing import build_model, wavegen
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


def sync(device: torch.device):
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    else:
        torch.cpu.synchronize(device)
# def get_losses():
#         loss = 0
#         return loss
def shuffle_utt(spk_embs, num_speakers, num_utterances):
    result = []
    for i in range(num_speakers):
        speaker_emb = spk_embs[i*num_utterances:i*num_utterances+ num_utterances,:]
        speaker_emb = speaker_emb[torch.randperm(num_utterances)]
        result.append(speaker_emb)
    result = torch.stack(result)
    return result

def trainning_procedure(config):
    writer = SummaryWriter(os.path.join(config['save_path'], 'logs'))
    dataset_path = os.path.join(config['data_path'])
    dataset_path = Path(dataset_path)
    dataset = SpeechDataset3(dataset_path, samples_length=64,
                            num_utterances=config['num_utterances'])
    loader = DataLoader(dataset, num_workers=8,
                        batch_size=config['num_speakers'],
                        pin_memory=True, shuffle=True, drop_last=True)

    
    speaker_emb = SpeakerEncoder(device, device)
    emb_ckt = torch.load(config['emb_model_path'])
    speaker_emb.load_state_dict(emb_ckt['model_state'])
    generator = Generator(32, 256, 512, 32).to(device)
    model_params = list(generator.parameters()) + list(speaker_emb.parameters())
    optimizer = torch.optim.Adam(
        # model_params,
        generator.parameters(),
        lr=config['lr'],
        betas=(0.99, 0.999))
    speaker_emb_optimizer = torch.optim.Adam(
        speaker_emb.parameters(),
        lr=config['lr'],
        weight_decay=5e-4
    )
    init_step = 0
    if config['load_ckp']:
        ckp = torch.load(os.path.join(config['save_path'],'save_ckp','model_ckp_10000.pt'))
        optimizer.load_state_dict(ckp['optimizer'])
        init_step = ckp['iteration']
        generator.load_state_dict(ckp['model_state'])
    
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    step = init_step

    while True:
        for _, speakers_batch in enumerate(loader, init_step):
 
            data = speakers_batch[0].type(torch.FloatTensor).to(device)
            data = data.view(-1, data.shape[-2], data.shape[-1])
            data = torch.transpose(data, -1, -2)
            sync(device)
            spk_embs = speaker_emb(data).to(device)
            # print('spk_emb shape:', spk_embs.shape)
            # emb_loss = spk_embs.view((config['num_speakers'], config['num_utterances'], -1)).to(device)
            # loss_emb,_ = speaker_emb.loss(emb_loss)

            ################### Shuffle utterances belong to a speaker #####################
            shuffle_spk_embs = shuffle_utt(spk_embs, config['num_speakers'], config['num_utterances'])
            shuffle_spk_embs = shuffle_spk_embs.view(-1, 256)
            ##############################################################################
            # print('shuffle_spk_embs shape: ', shuffle_spk_embs.shape)
            sync(device)

            mel_outputs, mel_outputs_posnet, encoder_out = generator(
                data, c_org=spk_embs, c_trg=spk_embs)
            mel_outputs_posnet = mel_outputs_posnet.squeeze(1)
            mel_outputs = mel_outputs.squeeze(1)

            _, _, encoder_out_hat = generator(
            mel_outputs_posnet, spk_embs, c_trg=spk_embs)

            loss_content = torch.nn.functional.l1_loss(
                encoder_out, encoder_out_hat)
            loss_recon = torch.nn.functional.mse_loss(data, mel_outputs_posnet)
            loss_recon_zero = torch.nn.functional.mse_loss(data, mel_outputs)
            optimizer.zero_grad()
            speaker_emb_optimizer.zero_grad()
            loss = loss_content + loss_recon + loss_recon_zero 
            sync(device)


            # speaker_emb.similarity_weight.retain_grad()
            # speaker_emb.similarity_bias.retain_grad()
            # speaker_emb.zero_grad()
            # loss_emb.backward()
            loss.backward()
            optimizer.step()
            # speaker_emb_optimizer.step()
            ########## write log #############################3
            if step % 1000:
                writer.add_scalar('Loss\Reconstruction Loss', loss_recon, step)
                writer.add_scalar('Loss\Reconstruction Loss Zero', loss_recon_zero, step)
                writer.add_scalar('Loss\Content Loss', loss_content, step)
                # writer.add_scalar('Loss\Speaker Embedding Loss', loss_emb, step)
            ##################################################
            
            print('iteration:{}-----Loss: {}'.format(step, loss.item()))
            if math.isnan(loss.item()):
                print('broken input data')
                continue
            
            step = step +1
            if step % config['save_iter'] == 0:
                print('saving model------------------------------')
                torch.save({
                    'iteration': step,
                    'model_state': generator.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, 
                os.path.join(config['save_path'], 'save_ckp','model_ckp_'+str(step)+'.pt'))
                convert_voice(loader, config, step)
            if step == 500000:
                steplr.step()


def convert_voice(loader, config, step):
    
    plt_path = os.path.join(config['save_path'], 'convert_voice')
    batch = next(iter(loader))
    data = batch[0].type(torch.FloatTensor).to(device)
    rnd_choice = np.random.choice(4,2,replace=False)
################ load data and init model #######################################
    src_mel = data[rnd_choice[0]]
    src_mel = torch.transpose(src_mel, -1, -2)
    trg_mel = data[rnd_choice[1]]
    trg_mel = torch.transpose(trg_mel, -1, -2)
 
    speaker_emb = SpeakerEncoder(device, device).to(device)
    generator = Generator(32, 256, 512, 32).to(device)
    generator_ckp = torch.load(os.path.join(config['save_path'],'save_ckp','model_ckp_'+str(step)+'.pt'))
    emb_ckp = torch.load(config['emb_model_path'])

    speaker_emb.load_state_dict(emb_ckp['model_state'])
    # speaker_emb = speaker_emb.to(device)
    generator.load_state_dict(generator_ckp['model_state'])
    # generator = generator.to(device)
##################################################################################

    src_identity = speaker_emb(src_mel).to(device)
    trg_identity = speaker_emb(trg_mel).to(device)

    _, mel_outputs_postnet,_ = generator(src_mel, src_identity, trg_identity)
    mel_outputs_postnet = mel_outputs_postnet.squeeze(1) 
    for i in range(5):

        origin_mel = src_mel[i]
        converted_mel = mel_outputs_postnet[i]

        plt.figure()
        plt.title('reconstructed mel spectrogram')
        librosa.display.specshow(converted_mel.cpu().detach().numpy(), x_axis='time', y_axis='mel', sr=16000)
        plt.colorbar(format='%f')
        plt.savefig(plt_path + '/convert_' + str(step)+'_utterance_'+str(i)+'.png')

        plt.figure()
        plt.title('original mel spectrogram')
        librosa.display.specshow(origin_mel.cpu().detach().numpy(), x_axis='time', y_axis='mel', sr=16000)
        plt.colorbar(format='%f')
        plt.savefig(plt_path + '/origin_' + str(step)+'_utterance_'+str(i)+'.png')

def convert_voice_wav(config):
        # dataset_path = os.path.join(config['data_path'])
        # dataset = SpeechDataset3(dataset_path, samples_length=192, num_utterances=50)
        # loader = DataLoader(dataset, num_workers=8, batch_size=4,
        #                     pin_memory=True, shuffle=True, drop_last=True)

        src_fp = '/home/ubuntu/VCTK-Corpus/new_encoder3/VCTK-Corpus_wav16_chunking_anh_hung/trim_16k_anh_hung_cut.npy_9.npy'
        trg_fp = '/home/ubuntu/VCTK-Corpus/new_encoder3/VCTK-Corpus_wav16_chunking_anh_vuong/trim_16k_anh_V_cut7.npy_9.npy'
        src_mel = np.load(src_fp, allow_pickle=True)
        trg_mel = np.load(trg_fp, allow_pickle=True)

        wav_path = os.path.join(config['save_path'], 'convert_voice', 'wav')
        if not os.path.exists(wav_path):
            os.mkdir(wav_path)
        plt_path = os.path.join(config['save_path'], 'convert_voice', 'plot')
        if not os.path.exists(plt_path):
            os.mkdir(plt_path)

        # batch = next(iter(loader))
        # data = batch[0].type(torch.FloatTensor).to(device)
        # rnd_choice = np.random.choice(4,2,replace=False)
        src_mel = torch.from_numpy(src_mel).unsqueeze(0).type(torch.FloatTensor).to(device)
        trg_mel = torch.from_numpy(trg_mel).unsqueeze(0).type(torch.FloatTensor).to(device)
        
        '''
        shape of batch is (data, utterances_id, speakers_id)
        speaker id shape is (utterance shape, speakers shape)
        utterance id shape is (utterance shape, speaker shape)
        '''
        # src_mel = data[rnd_choice[0]]
        # src_speaker_id = batch[2][0][rnd_choice[0]]
        src_mel = torch.transpose(src_mel, -1, -2)[:,0:192,:]
        # print('src mel shape: ', src_mel.shape)
        

        # trg_mel = data[rnd_choice[1]]
        trg_mel = torch.transpose(trg_mel, -1, -2)[:,0:192,:]
        # trg_speaker_id = batch[2][0][rnd_choice[1]]

        ###### Load model from ckp file ##################################
        speaker_emb = SpeakerEncoder(device, device).to(device)
        emb_ckt = torch.load(config['emb_model_path'])
        speaker_emb.load_state_dict(emb_ckt['model_state'])
        generator = Generator(32, 256, 512, 32).to(device)
        ckp = torch.load(os.path.join(config['save_path'],'save_ckp','model_ckp_150000.pt'))
        init_step = ckp['iteration']
        generator.load_state_dict(ckp['model_state'])

        vocoder_model = build_model().to(device)
        ckpt = torch.load('/home/ubuntu/checkpoint_step001000000_ema.pth')
        vocoder_model.load_state_dict(ckpt['state_dict'])
        ##############################################################
        src_identity = speaker_emb(src_mel).to(device)
        trg_identity = speaker_emb(trg_mel).to(device)

        _, mel_outputs_postnet,_ = generator(src_mel, src_identity, trg_identity)
        mel_outputs_postnet = mel_outputs_postnet.squeeze(1)

        for i in range(1):
            
            # src_utterance_id = batch[1][i][rnd_choice[0]].split('/')
            # src_utterance_id = src_utterance_id[-1].split('.')
            # src_utterance_id = src_utterance_id[0]

            # trg_utterance_id = batch[1][i][rnd_choice[1]].split('/')
            # trg_utterance_id = trg_utterance_id[-1].split('.')
            # trg_utterance_id = trg_utterance_id[0]
            src_utterance_id = 'anh_hung'
            trg_utterance_id = 'anh_vuong'

            fn = src_utterance_id + '_to_' + trg_utterance_id
            print('converting '+src_utterance_id+  ' to ' + trg_utterance_id)
            origin_mel = src_mel[i]
            converted_mel = mel_outputs_postnet[i]

            converted_mel = converted_mel.detach().cpu().numpy()
            origin_mel = origin_mel.detach().cpu().numpy()

            plt.figure()
            plt.title('reconstructed mel spectrogram')
            librosa.display.specshow(converted_mel, x_axis='time', y_axis='mel', sr=16000)
            plt.colorbar(format='%f')
            plt.savefig(plt_path + '/convert_' + fn +'.png')

            plt.figure()
            plt.title('original mel spectrogram')
            librosa.display.specshow(origin_mel, x_axis='time', y_axis='mel', sr=16000)
            plt.colorbar(format='%f')
            plt.savefig(plt_path + '/origin_' + fn +'.png')

            waveform = wavegen(vocoder_model, converted_mel)
            librosa.output.write_wav(os.path.join(wav_path, fn + '.wav'), waveform, sr=16000)

            

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', default='', type=str)
    parse.add_argument('--speakers_per_batch', default=4, type=int)
    parse.add_argument('--utterances_per_batch', default=50, type=int)
    parse.add_argument('--lr', default=1e-3, type=float)
    parse.add_argument('--epoches', default=100, type=int)
    parse.add_argument('--emb_model_path', default='', type=str)
    parse.add_argument('--save_iter', default=10000, type=int)
    # parse.add_argument('--ckp_path', default='/home/ubuntu/save_ckp/', type=str)
    parse.add_argument('--load_ckp', default=False, type=bool)
    parse.add_argument('--save_path', default='/home/ubuntu/autovc_log', type=str)
    args = parse.parse_args()

    config = dict(
        data_path=args.data_path,
        emb_model_path=args.emb_model_path,
        num_speakers=args.speakers_per_batch,
        num_utterances=args.utterances_per_batch,
        lr=args.lr,
        epoches=args.epoches,
        save_iter=args.save_iter,
        # ckp_path=args.ckp_path,
        load_ckp=args.load_ckp,
        save_path=args.save_path,
    )
    trainning_procedure(config)
    #convert_voice_wav(config)
    
