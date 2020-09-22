import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
import glob
# import preprocessing.utils as utils
import math
import pyworld as pw
import glob



fp = '/home/ubuntu/vcc2018_training/VCC2SF1/log_f0_VCC2SF1.npz'
fp_mcep = '/home/ubuntu/vcc2018_training/VCC2SF1/mcep_VCC2SF1.npz'
logf0 = np.load(fp)
mcep = np.load(fp_mcep)
logf0_mean = logf0['mean']
mcep_mean = mcep['mean']

print('logf0 mean: ', logf0_mean)
print('mcep mean: ', mcep_mean)
print('mcep mean shape: ', mcep_mean.shape)
# mel_spectro = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128, fmax=8000)
# mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectro), n_mfcc=36)



# recons_wav = librosa.feature.inverse.mfcc_to_audio(mfcc, n_mels=36)

# librosa.output.write_wav('/home/ubuntu/SM1_100002.wav', recons_wav, sr)