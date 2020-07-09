import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from random import randint
from pathlib import Path
import os
import soundfile as sf

from functools import wraps
from time import time
from tqdm import tqdm
import librosa

WINDOW_SIZE = 400
SHIFT_SIZE = 160

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('Elapsed time: {}'.format(end-start))
        return result
    return wrapper

class SpeechDataset(Dataset):
    def __init__(self, file_path, sr=16000, sample_frames=16000, num_utterances=10):
        self.file_path = file_path
        self.sr = sr
        self.sample_frames = sample_frames
        self.utterance_id = {}
        self.num_utterances = num_utterances
        self.speaker_ids = [f for f in (os.listdir(os.path.join(file_path)))]
        for i in range(len(self.speaker_ids)):
            file_name = [f for f in os.listdir(os.path.join(file_path, self.speaker_ids[i])) if os.path.isfile(os.path.join(file_path, self.speaker_ids[i], f))]
            self.utterance_id[self.speaker_ids[i]] = file_name

    def __len__(self):
        return len(self.speaker_ids)

    def __getitem__(self, index):
        speaker_id = self.speaker_ids[index]
        folder_path = os.path.join(self.file_path, speaker_id)
        utterances = self.utterance_id[speaker_id]
        rd_utterance_idx = np.random.choice(len(utterances), 1)
        file_name = os.path.join(folder_path, utterances[rd_utterance_idx[0]])
        audio, sr = sf.read(file_name, dtype='float32')
        # mel_specaa = librosa.feature.melspectrogram(audio, sr=16000)
        # print('mel spec shape: ', mel_specaa.shape)
        # print('length audio: ', len(audio))
        # rd_begin = np.random.choice((len(audio)-8000),1)[0]
        if len(audio)-32000 < 0:
                while len(audio) - 32000 < 0:
                    utterance_id_new = np.random.choice(len(utterances), 1)[0]
                    file_name = os.path.join(folder_path, utterances[utterance_id_new])
                    audio, sr = sf.read(file_name, dtype='float32')
        rd_begin = np.random.choice((len(audio)-32000),1)[0]
        time_data = []
        time_freq_data = []
        labels = []
        for i in range(self.num_utterances):
            
            
            # print('rd_idx: ',rd_begin)
            sample = audio[rd_begin:rd_begin + WINDOW_SIZE]
            # sample_time_freq = librosa.feature.melspectrogram(sample, sr=16000)
            # rd_begin+= SHIFT_SIZE
            rd_begin += WINDOW_SIZE
            # audio = torchvision.transforms.ToTensor(audio)
            time_data.append(sample)
            # time_freq_data.append(sample_time_freq)
            labels.append(speaker_id)
        time_data = torch.tensor(time_data)
        # time_freq_data = torch.tensor(time_freq_data)
        # return time_data, time_freq_data,labels, speaker_id
        return time_data, labels, speaker_id
# @timing
# def load_batch(loader):
#     # data, labels = next(iter(loader))
#     # print(labels)
#     # print(data.shape)
#     # return data, labels
#     for data, labels, speaker_id in iter(loader):
#         print(speaker_id)



# file_path = '/home/manhlt/extra_disk/VCTK-Corpus/wav16'
# dataset = SpeechDataset(file_path)
# dataloader = DataLoader(dataset, batch_size=4,
#                         shuffle=True, num_workers=0,
#                         pin_memory=True,drop_last=True)

# data, mel_spec,labels, speaker_id = next(iter(dataloader))
# print(mel_spec[0][0])
# load_batch(dataloader)

# for i in range(4):
#     load_batch(dataloader)
