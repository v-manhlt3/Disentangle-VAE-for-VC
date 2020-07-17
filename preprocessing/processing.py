import numpy as np 
import os, glob
import librosa

import torch
from tqdm import tqdm
import librosa
import librosa.display
from preprocessing.hparams import hparams
#from hparams import hparams
from wavenet_vocoder import builder
import matplotlib.pyplot as plt
import preprocessing.utils as utils
#import utils as utils

torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def build_model():
    
    model = getattr(builder, hparams.builder)(
        out_channels=hparams.out_channels,
        layers=hparams.layers,
        stacks=hparams.stacks,
        residual_channels=hparams.residual_channels,
        gate_channels=hparams.gate_channels,
        skip_out_channels=hparams.skip_out_channels,
        cin_channels=hparams.cin_channels,
        gin_channels=hparams.gin_channels,
        weight_normalization=hparams.weight_normalization,
        n_speakers=hparams.n_speakers,
        dropout=hparams.dropout,
        kernel_size=hparams.kernel_size,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_scales=hparams.upsample_scales,
        freq_axis_kernel_size=hparams.freq_axis_kernel_size,
        scalar_input=True,
        legacy=hparams.legacy,
    )
    return model



def wavegen(model, c=None, tqdm=tqdm):
    """Generate waveform samples by WaveNet.
    
    """

    model.eval()
    model.make_generation_fast_()

    Tc = c.shape[0]
    upsample_factor = hparams.hop_size
    # Overwrite length according to feature size
    length = Tc * upsample_factor

    # B x C x T
    c = torch.FloatTensor(c.T).unsqueeze(0)

    initial_input = torch.zeros(1, 1, 1).fill_(0.0)

    # Transform data to GPU
    initial_input = initial_input.to(device)
    c = None if c is None else c.to(device)

    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=None, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=hparams.log_scale_min)

    y_hat = y_hat.view(-1).cpu().data.numpy()

    return y_hat

def del_valid_data(fp):
    dataset_fp = os.path.join(fp)
    speakers_id = [speaker_id for speaker_id in os.listdir(dataset_fp)]
    # print(speakers_id)
    count = 0
    total_mel = 0 
    for speaker in speakers_id:
        speaker_fp = os.path.join(dataset_fp, speaker)
        # utterances = [utt for utt in os.listdir(speaker_fp) if os.path.isfile(os.path.join(speaker_fp, utt))]
        utterances = glob.glob(os.path.join(speaker_fp, '*.npy'))
        # print(utterances)
        for utt in utterances:
            mel = np.load(utt, allow_pickle=True)
            # print(mel.shape)
            total_mel = total_mel + 1
            if mel.shape[0] < 160:
                count= count+1
    print('The number of utt is lower than 2s: ', count)
    print('The total number of utt is: ', total_mel)

def vocoder(mel_spectrogram_fp, ckpt_fp):
    wav_fp = '/home/ubuntu/VCTK-Corpus/wav16/p244/p244_424.wav'
    print('torch device: ',device)
    vocoder_model = build_model().to(device)
    ckpt = torch.load(ckpt_fp)
    vocoder_model.load_state_dict(ckpt['state_dict'])
    mel_name = mel_spectrogram_fp.split("/")[-1]

    mel_spectrogram = np.load(mel_spectrogram_fp)
    wav, sr = librosa.load(wav_fp)
    mel_spectrogram2 = utils.melspectrogram(wav)

    print('mel_spectrogram file shape: ', mel_spectrogram.shape)
    print('mel_spectrogram from wav shape: ', mel_spectrogram2.shape)

    waveform = wavegen(vocoder_model, mel_spectrogram)

    librosa.output.write_wav('/home/ubuntu/'+mel_name+'_vocoder.wav', waveform, sr=16000)

def vocoder2(wav_fp, ckpt_fp):
    print('torch device: ',device)
    vocoder_model = build_model().to(device)
    ckpt = torch.load(ckpt_fp)
    vocoder_model.load_state_dict(ckpt['state_dict'])
    mel_name = wav_fp.split("/")[-1]

    wav, sr = librosa.load(wav_fp)
    # wav = wav[1000: 33768]
    mel_spectrogram = utils.melspectrogram(wav)
    mel_spectrogram = np.transpose(mel_spectrogram, (1, 0))
    print(mel_spectrogram.shape)

    waveform = wavegen(vocoder_model, mel_spectrogram)

    librosa.output.write_wav('/home/ubuntu/'+mel_name+'.wav', waveform, sr=16000)


def simple_inverse(mel_spectrogram_fp):
    mel_spectrogram = np.load(mel_spectrogram_fp)
    filter_mel = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=80, fmin=90)

    s_inv = librosa.feature.inverse.mel_to_stft(filter_mel)
    waveform = librosa.griffinlim(s_inv, n_iter=32, hop_length=256)

    librosa.output.write_wav('/home/ubuntu/'+'simple.wav', waveform, sr=16000)

if __name__=='__main__':
    fp = '/home/ubuntu/VCTK-Corpus/encoder'
    mel_fp = '/home/ubuntu/VCTK-Corpus/new_encoder/VCTK-Corpus_wav16_p376/p376_295.npy'
    wav_fp = '/home/ubuntu/VCTK-Corpus/wav16/p244/p244_424.wav'
    mel = np.load(mel_fp)

    plt.figure()
    plt.title('reconstructed mel spectrogram')
    librosa.display.specshow(mel, x_axis='time', y_axis='mel', sr=16000)
    plt.colorbar(format='%f')
    plt.savefig('/home/ubuntu/test_normalize_vctk.png')

    #simple_inverse(mel_fp)
    vocoder(mel_fp, '/home/ubuntu/checkpoint_step001000000_ema.pth')
    # vocoder2(wav_fp, '/home/ubuntu/checkpoint_step001000000_ema.pth')
    # del_valid_data(fp)
