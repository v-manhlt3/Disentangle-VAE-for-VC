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
import pickle

# WINDOW_SIZE = 400
# SHIFT_SIZE = 160

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('Elapsed time: {}'.format(end-start))
        return result
    return wrapper

class SpeechDataset1(Dataset):
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
        
        data = []
        utterance_ids = []
        for i in range(self.num_utterances):
            rd_utterance_idx = np.random.choice(len(utterances), 1)
            file_name = os.path.join(folder_path, utterances[rd_utterance_idx[0]])
            audio, sr = sf.read(file_name, dtype='float32')
            if len(audio)- self.sample_frames < 0:
                while len(audio) - self.sample_frames < 0:
                    utterance_id_new = np.random.choice(len(utterances), 1)[0]
                    file_name = os.path.join(folder_path, utterances[utterance_id_new])
                    audio, sr = sf.read(file_name, dtype='float32')
            rd_begin = np.random.choice((len(audio)- self.sample_frames),1)[0]
            audio = audio[rd_begin:rd_begin + self.sample_frames]
            # print('rd_idx: ',rd_begin)
            mel_spectrogram = librosa.feature.melspectrogram(audio, sr=self.sr,
                                                            n_fft=1024,hop_length=200,
                                                            win_length=800, n_mels=80)
            mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
            data.append(mel_spectrogram)
            utterance_ids.append(utterances[rd_utterance_idx[0]])

        data = torch.tensor(data)
        
        return data, utterance_ids, speaker_id

class SpeechDataset2(Dataset):
    def __init__(self, file_path, sr=16000, sample_frames=81, num_utterances=10):
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
        
        data = []
        utterance_ids = []
        for i in range(self.num_utterances):
            # print('Speaker id: ', speaker_id)
            rd_utterance_idx = np.random.choice(len(utterances), 1)
            file_name = os.path.join(folder_path, utterances[rd_utterance_idx[0]])
            # audio, sr = sf.read(file_name, dtype='float32')
            file = open(file_name, 'rb')
            sample = pickle.load(file)
            # print(sample.shape[1])
            file.close() 
            if sample.shape[1] - self.sample_frames < 0:
                while sample.shape[1] - self.sample_frames < 0:
                    utterance_id_new = np.random.choice(len(utterances), 1)[0]
                    file_name = os.path.join(folder_path, utterances[utterance_id_new])
                    file = open(file_name, 'rb')
                    sample = pickle.load(file)
                    file.close()
            rd_begin = np.random.choice((sample.shape[1] - self.sample_frames),1)[0]
            mel_spectrogram = sample[:, rd_begin:rd_begin + self.sample_frames]
            data.append(mel_spectrogram)
            utterance_ids.append(utterances[rd_utterance_idx[0]])

        data = torch.tensor(data)
        
        return data, utterance_ids, speaker_id
@timing
def load_batch(loader):
    # data, _,_ = next(iter(dataloader))
    for data,_,_ in iter(loader):
        # print()
        print(data.shape)




# import matplotlib.pyplot as plt
# import librosa.display

# file_path = '/home/manhlt/extra_disk/VCTK-Corpus/Mel_spectrogram'
# file_path2 = '/home/manhlt/extra_disk/VCTK-Corpus/wav16'
# dataset2 = SpeechDataset1(file_path2)
# dataset = SpeechDataset2(file_path)
# dataloader = DataLoader(dataset2, batch_size=4,
#                         shuffle=True, num_workers=0,
#                         pin_memory=True,drop_last=True)

# data, _,_ = next(iter(dataloader))
# print(data.shape)

# load_batch(dataloader)

# # mel = librosa.power_to_db(data[0][0].detach().numpy(), ref=np.max)
# librosa.display.specshow(data[0][0].detach().numpy(), x_axis='time', y_axis='log', sr=16000)
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel-frequency spectrogram')
# plt.tight_layout()
# plt.show()