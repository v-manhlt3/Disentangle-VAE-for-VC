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



fp = '/home/ubuntu/vcc2016_training/SM1/100002.wav'
wav, sr = librosa.load(fp)
mel_spectro = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128, fmax=8000)
mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spectro), n_mfcc=36)


recons_wav = librosa.feature.inverse.mfcc_to_audio(mfcc, n_mels=36)

librosa.output.write_wav('/home/ubuntu/SM1_100002.wav', recons_wav, sr)