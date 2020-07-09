import torch
from torch.utils.data import DataLoader, Dataset
from preprocessing.encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from pathlib import Path
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


def trainning_procedure(config):
    dataset_path = os.path.join(config['data_path'])
    dataset_path = Path(dataset_path)
    dataset = SpeakerVerificationDataset(dataset_path)
    loader = SpeakerVerificationDataLoader(
        dataset,
        num_workers=8,
        speakers_per_batch=2,
        utterances_per_speaker=50, pin_memory=True)

    
    speaker_emb = SpeakerEncoder(device, device)
    emb_ckt = torch.load(config['emb_model_path'])
    speaker_emb.load_state_dict(emb_ckt['model_state'])
#    generator = Generator(32, 256, 512, 32).to(device)
    generator = Generator(16, 256, 512, 16).to(device)
    optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=config['lr'],
        betas=(0.99, 0.999))
    init_step = 0
    if config['load_ckp']:
        ckp = torch.load(os.path.join(config['save_path'],'save_ckp','model_ckp_210000.pt'))
        #ckp = torch.load(os.path.join('/home/ubuntu/autovc_log','save_ckp','model_ckp_140000.pt'))
        optimizer.load_state_dict(ckp['optimizer'])
        init_step = ckp['iteration']
        generator.load_state_dict(ckp['model_state'])
    
    steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    print("trainingggggggggggggggg")
    
    # X = torch.FloatTensor(config['num_speakers']*config['num_utterances'], 80, 63).to(device)
    for step, speakers_batch in enumerate(loader, init_step):
        data = torch.from_numpy(speakers_batch.data).to(device)
        sync(device)
        spk_embs = speaker_emb(data).to(device)
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
        loss = loss_content + loss_recon + loss_recon_zero
        sync(device)
        
        print('iteration:{}-----Loss: {}'.format(step, loss.item()))
        if math.isnan(loss.item()):
            print('broken input data')
            continue
        loss.backward()
        optimizer.step()
        if step % config['save_iter'] == 0:
            print('saving model------------------------------')
            torch.save({
                'iteration': step,
                'model_state': generator.state_dict(),
                'optimizer': optimizer.state_dict()
            }, 
            os.path.join(config['save_path'], 'save_ckp','model_ckp_'+str(step)+'.pt'))
            convert_voice(loader, config, step)
        if step == 50000:
            steplr.step()


def convert_voice(loader, config, step):
    plt_path = os.path.join(config['save_path'], 'convert_voice')
    batch = next(iter(loader))
    data = torch.from_numpy(batch.data).to(device)
    rnd_choice = np.random.choice(4,2,replace=False)
################ load data and init model #######################################
    src_mel = data[0:50,:,:]
    # src_mel = torch.from_numpy(src_mel).to(device)
    trg_mel = data[50:100,:,:]
    # trg_utt = torch.from_numpy(trg_utt).to(device)

    speaker_emb = SpeakerEncoder(device, device).to(device)
    #generator = Generator(32, 256, 512, 32).to(device)
    generator = Generator(16, 256, 512, 16).to(device)
    #print('loading data from file path: '+config['save_path'])
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
    # print('batch data shape: ', data.shape)
    # print('src_mel shape: ', src_mel.shape)
    # print('mel outputs postnet shape: ', mel_outputs_postnet.shape)    
    for i in range(5):

        origin_mel = src_mel[i]
        converted_mel = mel_outputs_postnet[i]
        # print('mel_origin shape: ', origin_mel.shape)
        # print('converted_mel shape: ', converted_mel.shape)

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


if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--data_path', default='', type=str)
    parse.add_argument('--speakers_per_batch', default=64, type=int)
    parse.add_argument('--utterances_per_batch', default=10, type=int)
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

    # clean_data_root = '/home/ubuntu/LibriSpeech/mel_spectrogram/'
    # filename2 = '/home/ubuntu/testtt.png'
    # clean_data_root = Path(clean_data_root)
    # dataset = SpeakerVerificationDataset(clean_data_root)
    # loader = SpeakerVerificationDataLoader(
    #         dataset,
    #         64,
    #         10,
    #         num_workers=8,)

    # speaker_batch = next(iter(loader))
    # print(speaker_batch.data.shape)
    # data = speaker_batch.data[0]
    # plt.figure()
    # plt.title('original mel spectrogram')
    # librosa.display.specshow(data, x_axis='time', y_axis='mel', sr=16000)
    # plt.colorbar(format='%f')
    # plt.savefig(filename2)
