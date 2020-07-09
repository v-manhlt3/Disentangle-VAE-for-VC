import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import matplotlib.pyplot as plt

from autovc import Encoder, Decoder, Postnet, Classifier
from tensorboardX import SummaryWriter
import numpy as np
import os
import soundfile as sf
import timeit
from torch.utils.data import DataLoader
# from speech_preprocessing.dataloader import SpeechDataset
from preprocessing.dataset_mel import SpeechDataset, speaker_to_onehot, SpeechDataset2
import vanila_autoencoder
from speaker_verfi_model import Model, calc_loss, get_centroids
from preprocessing.vocoder2waveform import build_model, wavegen
import librosa
import librosa.display
from torch.optim.lr_scheduler import StepLR

torch.autograd.set_detect_anomaly(True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.004
#device = torch.device('cpu')

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
### tile the output n times ##################################
def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def accumulate_evidence(mu, logvar, is_cuda, batch_labels):
    var_dict = {}
    mu_dict = {}

    # convert logvar to variance for calculations
    var = logvar.exp_()

    # calculation var inverse for each group using group vars
    # print('lengt of batch_labels: ', len(batch_labels))
    for i in range(len(batch_labels)):
        group_label = batch_labels[i]

        # remove 0 value from variances 
        for j in range(len(logvar[0])):
            if var[i][j] ==0:
                var[i][j] = 1e-6
        if group_label in var_dict.keys():
            var_dict[group_label]+= (1/var[i])
        else:
            var_dict[group_label] = (1/var[i])

    # invert var inverses to calculate mu and return value
    for group_label in var_dict.keys():
        var_dict[group_label] = 1 / var_dict[group_label]
    
    # calculate mu for each group
    for i in range(len(batch_labels)):
        group_label = batch_labels[i]

        if group_label in mu_dict.keys():
            mu_dict[group_label] += mu[i]*(1/logvar[i])
        else:
            mu_dict[group_label] = mu[i]*(1/logvar[i])

    # multply group var with sums calculated above to get mu for the group
    for group_label in mu_dict.keys():
        mu_dict[group_label] *= var_dict[group_label]
        
    group_mu = torch.FloatTensor(len(mu_dict), mu.shape[1])
    group_var =  torch.FloatTensor(len(var_dict), var.shape[1])

    if is_cuda:
        group_mu.cuda()
        group_var.cuda()

    idx = 0
    for key in var_dict.keys():
        group_mu[idx] = mu_dict[key]
        group_var[idx] = var_dict[key]

        for j in range(len(group_var[idx])):
            if group_var[idx][j] == 0:
                group_var[idx][j] = 1e-6

        idx=idx+1

    return Variable(group_mu, requires_grad=True), Variable(torch.log(group_var), requires_grad=True)



def training_procedure(config):
    save_path = os.path.join(config['log_path'],'checkpoint')
    # print(config.batch_size)
    enc_cfg = config['enc_config']
    dec_cfg = config['dec_config']
    # dataset = SpeechDataset(config['file_path'], num_utterances=config['num_utterance'],samples_length=16384)
    dataset = SpeechDataset2(config['file_path'], num_utterances=config['num_utterance'])
    loader = DataLoader(dataset, batch_size=config['batch_size'],
                        shuffle=True, num_workers=0,
                        pin_memory=True,drop_last=True)

    ############## vae model ############################
    if config['model'] == 'autoencoder':
        encoder = vanila_autoencoder.Encoder(256, 64, 80)
        decoder = vanila_autoencoder.Decoder(256, 80, 67)
        postnet = Postnet()
        cluster = Model(10, 20).to(device)
        classifier = Classifier(config['latent_dim'], num_speaker=109).to(device)
    ############## autovc vae model ####################
    else:
        encoder = Encoder(256, 64,80)
        decoder = Decoder(256, 80, 67)
        postnet = Postnet()
        classifier = Classifier(config['latent_dim'], num_speaker=109).to(device)
        cluster = Model(10, 20).to(device)

    if config['load_model']:
        print("---------load pretrained model-----------")

    
    """"Variable definition"""
    X = torch.cuda.FloatTensor(config['batch_size']*config['num_utterance'], 80, 67)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    postnet = postnet.to(device)
    X = X.to(device)
    
    optimizer = Adam(
        list(encoder.parameters())+list(decoder.parameters()),
        lr=config['lr'],
        betas=(config['beta_1'], config['beta_2'])
    )

    steplr = StepLR(optimizer, step_size=1, gamma=0.5)
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=60, min_lr=8e-5,verbose=True, factor=0.5)

    writer = SummaryWriter(os.path.join(config['log_path'], 'tensorboard_dir'))
    last_recons_loss = 0
    count_lr_decay = 0 
    for epoch in range(config['epoch']):
        recon_loss_total = 0
        kl_style_loss_total = 0
        kl_content_loss_total = 0
        crossentropy_loss_total = 0
        distance_loss_total = 0

        print('Epoch #{}-----------------------------------------------------   --'.format(epoch))
        

        # for iteration in range(109//config['batch_size']):
        # for batch_data, batch_mel_spec, batch_labels, speaker_ids in iter(loader):
        for batch_data, batch_labels, speaker_ids in iter(loader):
            batch_data = batch_data.view(-1, 80, 67)
            batch_data.cuda()

            optimizer.zero_grad()

            X.copy_(batch_data)
            if config['model'] == 'vae':
                style_mu, style_logvar, content_mu, content_logvar = encoder(X)

                group_content_mu, group_content_logvar = accumulate_evidence(content_mu, content_logvar, True, speaker_ids)


                group_content_mu1 = tile(group_content_mu, 0, config['num_utterance'])
                group_content_logvar1 = tile(group_content_logvar, 0, config['num_utterance'])
                style_kl_loss = (-0.5)*torch.sum(1 + style_logvar - style_mu.pow(2) - style_logvar.exp())
                group_content_kl_loss = (-0.5)*torch.sum(1 + group_content_logvar -group_content_mu.pow(2) - group_content_logvar.exp())

                recons_audio = decoder(style_mu, style_logvar, group_content_mu1, group_content_logvar1)
            else:
                style, content = encoder(X)
                recons_audio = decoder(style, content)


            ### compute loss #########################################
            
            post_recons_audio = postnet(recons_audio)
            post_recons_audio = post_recons_audio + recons_audio
            post_recons_audio = torch.clamp(post_recons_audio, min=0, max=1)
    
            recons_loss1 = torch.nn.functional.l1_loss(recons_audio, X).to(device)
            recons_loss2 = torch.nn.functional.l1_loss(post_recons_audio, X).to(device)

            recons_loss = recons_loss1 + recons_loss2  

            ### train classifier to enforce encoder encodes speaker identity
            onehot_speaker = speaker_to_onehot(speaker_ids, dataset.speaker_ids, num_utterance=config['num_utterance']).to(device).long()
            # z_speaker = decoder.reparameterization(content_mu, content_logvar)
            prob_speaker = classifier(content_mu).float()
            onehot_speaker = onehot_speaker.squeeze_()
            crossentropy_loss = torch.nn.functional.cross_entropy(prob_speaker, onehot_speaker)
            crossentropy_loss_total += crossentropy_loss
            ##############################################################

            #### train a clusterior to instead cluster data######################
            # z_speaker = decoder.reparameterization(content_mu, content_logvar)
            # distance_loss = 0
            # for idx in range(content.shape[0]):
            #     centroids = get_centroids(config['batch_size'], config['num_utterance'], content, idx)
            #     simi_matrix = cluster(content, centroids)
            #     distance_loss = distance_loss + calc_loss(simi_matrix)
            # # distance_loss = distance_loss / z_speaker.shape[0]
            # distance_loss_total = distance_loss_total + distance_loss
            #####################################################################

            recon_loss_total += recons_loss
            

            if config['model'] =='vae':
                kl_style_loss_total +=style_kl_loss
                kl_content_loss_total += group_content_kl_loss

                style_kl_loss = style_kl_loss.to(device)
                group_content_kl_loss = group_content_kl_loss.to(device)
                loss = crossentropy_loss + recons_loss + config['kl_coef']*style_kl_loss/(config['num_utterance']*config['batch_size']) + (config['kl_coef']*group_content_kl_loss) / config['batch_size']
                # loss = recons_loss + config['kl_coef']*style_kl_loss/(config['num_utterance']*config['batch_size']) + (config['kl_coef']*group_content_kl_loss) / config['batch_size']
                
            else:
                loss = crossentropy_loss + recons_loss
            loss.backward()
            print('Loss: ', loss.item())
            optimizer.step() 

        ################# Decay learning rate a half ################################
        # if epoch > 15:
        #     if (last_recons_loss - recon_loss_total/(109//config['batch_size'])) <= THRESHOLD:
        #         count_lr_decay += 1
        #     else:
        #         count_lr_decay = 0
        # if count_lr_decay == 7:
        #     steplr.step()
        #     count_lr_decay = 0
        # last_recons_loss = recon_loss_total/(109//config['batch_size'])
        scheduler_lr.step(recon_loss_total)
        #############################################################################
        cur_lr = get_lr(optimizer)
        if config['model'] == 'vae':
            writer.add_scalar('tag/KL Style Loss', kl_style_loss_total, epoch)
            writer.add_scalar('tag/KL Content Loss', kl_content_loss_total, epoch)
        writer.add_scalar('tag/Reconstruction Loss', recon_loss_total/(109//config['batch_size']), epoch)
        writer.add_scalar('tag/Cross Entropy Loss', crossentropy_loss_total, epoch)
        writer.add_scalar('tag/Learning Rate', cur_lr, epoch)
        if (epoch+1) % 100 == 0:
            torch.save(encoder.state_dict(),os.path.join(save_path, 'autovc_encoder_'+str(epoch+1)+'.pkl'))
            torch.save(decoder.state_dict(), os.path.join(save_path, 'autovc_decoder_'+str(epoch+1)+'.pkl'))
            torch.save(postnet.state_dict(), os.path.join(save_path, 'autovc_postnet_'+str(epoch+1)+'.pkl'))
            torch.save(classifier.state_dict(), os.path.join(save_path, 'autovc_classifier_'+str(epoch+1)+'.pkl'))
            torch.save(cluster.state_dict(), os.path.join(save_path, 'autovc_cluster_'+str(epoch+1)+'.pkl'))

            inference(config, epoch, encoder, decoder, postnet, loader)
            encoder.train()
            decoder.train()
            postnet.train()

    torch.save(encoder.state_dict(),save_path + 'encoder_lastest.pkl')
    torch.save(decoder.state_dict(), save_path + 'decoder_lastest.pkl')
    torch.save(postnet.state_dict(), save_path + 'postnet_lastest.pkl')

def inference(config, epoch, encoder, decoder, postnet, loader):
    encoder.eval()
    decoder.eval()
    postnet.eval()

    vocoder_ckpt_path = '/vinai/manhlt/icassp-20/checkpoint_step001000000_ema.pth'
    model = build_model().to(device)
    checkpoint_vocoder = torch.load(vocoder_ckpt_path)
    model.load_state_dict(checkpoint_vocoder['state_dict'])

    ckp_path = config['log_path']
    sample_fp = config['log_path']
    enc_cfg = config['enc_config']
    dec_cfg = config['dec_config']
    # dataset = SpeechDataset(config['file_path'], num_utterances=config['num_utterance'])
    # loader = DataLoader(dataset, batch_size=config['batch_size'],
    #                     shuffle=True, num_workers=0,
    #                     pin_memory=True,drop_last=True)


    ############## vae model ############################
    # encoder = Encoder(128, 80)
    # decoder = Decoder(128, 80, 81)
    ############## autovc vae model ####################
    # encoder = Encoder(256, 32,80)
    # decoder = Decoder(256, 80, 131)
    # postnet = Postnet()
    X = torch.cuda.FloatTensor(config['batch_size']*config['num_utterance'], 80, 67)
    # encoder = encoder.to(device)
    # decoder = decoder.to(device)
    # postnet = postnet.to(device)
    # X = X.to(device)
    # if config['cuda']:
    #     encoder.cuda()
    #     decoder.cuda()
    #     X.cuda()

    # encoder.load_state_dict(
    #     torch.load(os.path.join(ckp_path,'autovc_encoder_'+str(epoch+1)+'.pkl'), map_location=lambda storage, loc: storage)
    # )
    # decoder.load_state_dict(
    #     torch.load(os.path.join(ckp_path,'autovc_decoder_'+str(epoch+1)+'.pkl'), map_location=lambda storage, loc:storage)
    # )
    # postnet.load_state_dict(
    #     torch.load(os.path.join(ckp_path,'autovc_postnet_'+str(epoch+1)+'.pkl'), map_location=lambda storage, loc:storage)
    # )

    batch_data, batch_labels, speaker_ids = next(iter(loader))
    batch_data = batch_data.view(-1, 80, 67)
    print('batch data shape: ', batch_data.shape)
    X.copy_(batch_data)
    for i in range(X.shape[0]):
        if config['model'] =='vae':
            style_mu, style_logvar, content_mu, content_logvar = encoder(X)
            recons_audio = decoder(style_mu, style_logvar, content_mu, content_logvar)
            post_mel = postnet(recons_audio)
            recons_audio = recons_audio + post_mel
        else:
            style, content = encoder(X)
            recons_audio = decoder(style, content)
            post_mel = postnet(recons_audio)
            recons_audio = recons_audio + post_mel
    fs = 16000
    for i in range(5):

        filename = os.path.join(sample_fp, 'inference',str(epoch)+'_sample_'+str(i)+'_.png')
        filename2 = os.path.join(sample_fp, 'inference',str(epoch)+'_original_'+str(i)+'_.png')
        
        mel_spec_recons = recons_audio[i]
        mel_spec_recons = torch.transpose(recons_audio[i], 1, 0)
       
        mel_spec_origin = batch_data[i]
        # mel_spec_recons = mel_spec_recons + post_mel
        mel_spec_origin = torch.transpose(batch_data[i], 1, 0)
        # reconstructed_audio = wavegen(model, mel_spec_recons)
        # origin_audio = wavegen(model, mel_spec_origin)
        ### plot mel spectrogram ##################################
        plt.figure()
        plt.title('reconstructed mel spectrogram')

        librosa.display.specshow(mel_spec_recons.cpu().detach().numpy(), x_axis='time', y_axis='mel', sr=16000)
        plt.colorbar(format='%f')
        plt.savefig(filename)
        # plt.plot(mel_spec_recons.detach().numpy())
        plt.figure()
        plt.title('original mel spectrogram')
        librosa.display.specshow(mel_spec_origin.cpu().detach().numpy(), x_axis='time', y_axis='mel', sr=16000)
        plt.colorbar(format='%f')
        plt.savefig(filename2)
        # plt.plot(mel_spec_origin.detach().numpy())
        # plt.show()

        # filename = os.path.join(sample_fp, 'sample_'+str(i)+'.wav')
        # filename2 = os.path.join(sample_fp, 'original_'+str(i)+'.wav')
        # librosa.output.write_wav(filename, reconstructed_audio, sr=16000)
        # librosa.output.write_wav(filename2, origin_audio, sr=16000)
        
        




if __name__ =='__main__':
    import argparse
    import json
    parse = argparse.ArgumentParser()
    #### Parse argument ################################################
    parse.add_argument('--file_path', default='/home/ubuntu/VCTK-Corpus/mel_spectrogram', type=str)
    parse.add_argument('--batch_size', default=2, type=int)
    parse.add_argument('--num_utterance', default=10, type=int)
    parse.add_argument('--lr', default=1e-3, type=float)
    parse.add_argument('--load_model', default=False, type=bool)
    parse.add_argument('--cuda', default=True, type=bool)
    parse.add_argument('--epoches', default=500, type=int)
    parse.add_argument('--latent_dim', default=256, type=int)
    parse.add_argument('--kl_coef', default=1.0, type=float)
    parse.add_argument('--log_path', default='/home/ubuntu/log', type=str)
    parse.add_argument('--note', default='', type=str)
    parse.add_argument('--model', default='vae', type=str)
    args = parse.parse_args()
    # batch_size = args.batch_size
    # print(batch_size)
    config = dict(
        file_path=args.file_path,
        batch_size=args.batch_size,
        num_utterance=args.num_utterance,
        load_model=args.load_model,
        cuda=args.cuda,
        epoch=args.epoches,
        kl_coef=args.kl_coef,
        lr=args.lr,
        log_path=args.log_path,
        beta_1=0.9,
        beta_2=0.99,
        latent_dim=args.latent_dim,
        enc_config=dict(
            in_dim=1,
            strides=[1,1,1],
            channels=[2,2,2],
            latent_dim=28,
            kernel_sizes=[3,3,3]
        ),
        dec_config=dict(
            in_dim=2,
            channels=[2,2,2],
            strides=[1,1,1],
            kernel_sizes=[3,3,3],
            latent_dim=28
        ),
        model=args.model,
        note=args.note
    )
    
    if not os.path.exists(os.path.join(args.log_path)):
        os.makedirs(os.path.join(args.log_path))
        os.makedirs(os.path.join(args.log_path, 'inference'))
        os.makedirs(os.path.join(args.log_path, 'checkpoint'))

    with open(os.path.join(args.log_path,'config.json'), 'w') as js_file:
        json.dump(config, js_file, indent=4, sort_keys=True)
    # loader = Audio_Loader(config['file_path'], config['batch_size'], config['num_utterance'])
    # inference(config)
    training_procedure(config)

