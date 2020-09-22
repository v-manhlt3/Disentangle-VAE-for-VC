import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np 
import librosa
import os
import preprocessing.utils as processing
# import utils as processing
from functools import wraps
from time import time
import glob
import pickle

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

def get_col_txt(fp, col_number):

    token = open(fp, 'r')
    linestoken = token.readlines()
    result = []

    for line in linestoken:
        result.append(line.split()[col_number]) 

    del result[0]
    return result

def get_male_spk(fp):
    rmv_idx = []
    fp_spk_info = os.path.join(fp, 'speaker-info.txt')
    file_header = ['ID', 'AGE', 'GENDER', 'ACCENTS', 'REGION']

    spk_gender = get_col_txt(fp_spk_info, file_header.index('GENDER'))
    spk_id = get_col_txt(fp_spk_info, file_header.index('ID'))
    assert len(spk_gender) == len(spk_id)
    for idx in range(len(spk_gender)):
        if spk_gender[idx] == 'F':
            rmv_idx.append(idx)
            
    spk_id = ['VCTK-Corpus_wav16_p'+spk for idx, spk in enumerate(spk_id) if idx not in rmv_idx]
    spk_id.append('VCTK-Corpus_wav16_chunking_barackobama')
    return spk_id


class SpeechDatasetGVAE(Dataset):
    def __init__(self, file_path,sr=16000, samples_length=64):
        self.file_path =file_path
        self.sr = sr
        self.samples_length = samples_length

        self.speaker_ids = [f for f in os.listdir(self.file_path)]
        self.spk_utt = []
        self.utterance_fp = np.array([])

        for i in range(len(self.speaker_ids)):
            spk_utt = np.array(glob.glob(os.path.join(self.file_path,self.speaker_ids[i], "*.npy")))
            np.random.shuffle(spk_utt)
            self.spk_utt.append(spk_utt)
            spk_utt1 = spk_utt[:spk_utt.shape[0]//2]
            spk_utt2 = spk_utt[spk_utt.shape[0]//2 : spk_utt.shape[0]//2 + spk_utt.shape[0]//2]
            spk_utt = [(spk_utt1[i],spk_utt2[i]) for i in range(len(spk_utt1))]
            spk_utt = np.array(spk_utt)
            # print('speaker id: ', self.speaker_ids[i])
            # print('speaker utterances: ', spk_utt.shape)
            if i == 0:
                self.utterance_fp = spk_utt
            else:
                self.utterance_fp = np.concatenate((self.utterance_fp, spk_utt), axis=0)

    def shuffle_data(self):
        self.utterance_fp = np.array([])

        for i in range(len(self.spk_utt)):
            utt = self.spk_utt[i] 
            np.random.shuffle(utt)
            utt1 = utt[:utt.shape[0]//2]
            utt2 = utt[utt.shape[0]//2 : utt.shape[0]//2 + utt.shape[0]//2]
            utt = [(utt1[i], utt2[i]) for i in range(utt1.shape[0])]
            utt = np.array(utt)
            if i == 0:
                self.utterance_fp = utt
            else:
                self.utterance_fp = np.concatenate((self.utterance_fp, utt), axis=0)

    def __getitem__(self, index):
        # print(self.utterance_fp[index])
        utterance1 = self.utterance_fp[index, 0]
        utterance2 = self.utterance_fp[index, 1]
        
        mel1 = np.load(utterance1)
        # print('mel1 shape: ', mel1.shape)
        # mel1 = mel1.T
        # print('mel1 transpose shape: ', mel1.shape)
        mel2 = np.load(utterance2)
        # mel2 = mel2.T
        
        if mel1.shape[1] < self.samples_length:
            mel1 = np.pad(mel1, ((0,0),(0,self.samples_length - mel1.shape[1])), 'constant', constant_values=0)
        else:
            rd_begin1 = np.random.choice((mel1.shape[1] - self.samples_length), 1)[0]
            mel1 = mel1[:,rd_begin1:rd_begin1 + self.samples_length]
        if mel2.shape[1] < self.samples_length:
            mel2 = np.pad(mel2, ((0,0),(0,self.samples_length - mel2.shape[1])), 'constant', constant_values=0)            
        else:
            rd_begin2 = np.random.choice((mel2.shape[1] - self.samples_length), 1)[0]    
            mel2 = mel2[:,rd_begin2:rd_begin2 + self.samples_length]

        # utterance_id = utterance1.split('/')[-1].split('.')[0]
        spk_id = utterance1.split('/')[-2]
        spk_id = self.speaker_ids.index(spk_id)
        # print('mel1 shape: ', mel1.shape)

        return torch.from_numpy(mel1), torch.from_numpy(mel2), torch.tensor(spk_id)

    def __len__(self):
        return len(self.utterance_fp)
    
    def get_utterance(self, speaker, utterance):

        fp = os.path.join(self.file_path, speaker, utterance)
        mel_spec = np.load(fp)
        return mel_spec

class SpeechDataset2(Dataset):
    def __init__(self, file_path,sr=16000, samples_length=64):
        self.file_path =file_path
        self.sr = sr
        self.samples_length = samples_length
        # self.num_utterances = num_utterances

        self.speaker_ids = [f for f in os.listdir(self.file_path)]
        self.utterance_fp = np.array([])
        for speaker in self.speaker_ids:
            # print(speaker)
            spk_utt = np.array(glob.glob(os.path.join(self.file_path,speaker, "*.npy")))
            self.utterance_fp = np.concatenate((self.utterance_fp, spk_utt), axis=None)

    def __getitem__(self, index):

        utterance = self.utterance_fp[index]
        mel_spec = np.load(utterance)
        # np.transpose(mel_spec, (-1, -2))

        if mel_spec.shape[1] < self.samples_length:
            mel_spec = np.pad(mel_spec, ((0,0),(0,self.samples_length-mel_spec.shape[1])), 'constant', constant_values=0)

        else:
            rd_begin = np.random.choice((mel_spec.shape[1] - self.samples_length), 1)[0]
            mel_spec = mel_spec[:,rd_begin:rd_begin + self.samples_length]

        # data.append(mel_spec)
        # utterances.append(utterance)
        # data = torch.tensor(data)
        # file_info = utterance.split('/')[-1]
        # file_info = file_info.split('.')[0]
        # file_info = file_info.split('_')
        utterance_id = utterance.split('/')[-1].split('.')[0]
        spk_id = utterance.split('/')[-2]
        spk_id = self.speaker_ids.index(spk_id)

        return torch.from_numpy(mel_spec), spk_id

    def __len__(self):
        return len(self.utterance_fp)

    def get_batch_utterances(self, speaker_id, num_utterances):

        spk_utt = np.array(glob.glob(os.path.join(self.file_path, speaker_id, "*.npy")))
        rnd_idx = np.random.choice(len(spk_utt), num_utterances)
        utterances = []
        data = []
        speaker_ids = []
        spk_id = self.speaker_ids.index(speaker_id)
        for i in range(num_utterances):
            # folder_path = os.path.join(self.file_path, speaker_id)
            rd_uttrance = np.random.choice(len(spk_utt), 1)[0]
            # utterance = self.utterance_ids[speaker_id][rd_uttrance]
            mel_spec = np.load(spk_utt[rd_uttrance])

            if mel_spec.shape[1] <= self.samples_length:
                while mel_spec.shape[1] < self.samples_length:

                    rd_uttrance = np.random.choice(len(spk_utt), 1)[0]
                    # utterance = self.utterance_ids[speaker_id][rd_uttrance]
                    mel_spec = np.load(spk_utt[rd_uttrance])

            rd_begin = np.random.choice((mel_spec.shape[1] - self.samples_length), 1)[0]
            mel_spec = mel_spec[:,rd_begin:rd_begin + self.samples_length]

            data.append(mel_spec)
            # utterances.append(utterance)
            speaker_ids.append(spk_id)

        data = torch.tensor(data)
        speaker_ids = torch.tensor(speaker_ids)
        return data, speaker_ids

    def get_batch_speaker(self, utterance): 

        data = []
        speaker_id = []
        utt = []
        for spk in self.speaker_ids:
            mel_spec = np.load(os.path.join(self.file_path, spk, utterance))
            print('mel shape: ', mel_spec.shape[1])
            rd_begin = np.random.choice((mel_spec.shape[1] - self.samples_length), 1)[0]
            mel_spec = mel_spec[:,rd_begin:rd_begin + self.samples_length]

            data.append(mel_spec)
            speaker_id.append(spk)
            utt.append(utterance)

        data = torch.tensor(data)
        return data, utt, speaker_id
    
    def get_utterance(self, speaker, utterance):

        fp = os.path.join(self.file_path, speaker, utterance)
        mel_spec = np.load(fp)
        return mel_spec

########## Dataset for loading mel from numpy file #####################################
class SpeechDataset3(Dataset):
    def __init__(self, file_path,sr=16000, samples_length=64, num_utterances=40, male_dataset=False):
        self.file_path =file_path
        self.sr = sr
        self.samples_length = samples_length
        self.num_utterances = num_utterances
        self.speaker_info_fp = os.path.join(self.file_path, 'speaker-info.txt')
        self.utterance_ids = {}

        if male_dataset:
            self.speaker_ids = get_male_spk(self.file_path)
        else:
            self.speaker_ids = [f for f in os.listdir(self.file_path)]
        
        for speaker in self.speaker_ids:
            self.utterance_ids[speaker] = glob.glob(os.path.join(self.file_path, speaker, '*.npy'))

    def __getitem__(self, index):
        speaker_id = self.speaker_ids[index]
        utterances = []
        data = []
        speaker_ids = []
        for i in range(self.num_utterances):
            folder_path = os.path.join(self.file_path, speaker_id)
            rd_uttrance = np.random.choice(len(self.utterance_ids[speaker_id]), 1)[0]
            utterance = self.utterance_ids[speaker_id][rd_uttrance]
            mel_spec = np.load(utterance)
            # print('load mel spec shape: ', mel_spec.shape)
            # transpose for new new_encoder2 dataset
            np.transpose(mel_spec, (-1, -2))
            if mel_spec.shape[1] <= self.samples_length:
                while mel_spec.shape[1] < self.samples_length:
                    rd_uttrance = np.random.choice(len(self.utterance_ids[speaker_id]), 1)[0]
                    utterance = self.utterance_ids[speaker_id][rd_uttrance]
                    mel_spec = np.load(utterance)

            rd_begin = np.random.choice((mel_spec.shape[1] - self.samples_length), 1)[0]
            mel_spec = mel_spec[:,rd_begin:rd_begin + self.samples_length]
            # print('load mel spec shape: ', mel_spec.shape)
            # print('spk id: ', speaker_id)
            data.append(mel_spec)
            utterances.append(utterance)
            speaker_ids.append(index)
        
        data = torch.tensor(data)
        speaker_ids = torch.tensor(speaker_ids)
        # print('data shape: ', data.shape)
        return data, utterances, speaker_ids


    def get_batch_utterances(self, speaker_id, num_utterances):

        spk_utt = np.array(glob.glob(os.path.join(self.file_path, speaker_id, "*.npy")))
        rnd_idx = np.random.choice(len(spk_utt), num_utterances)
        utterances = []
        data = []
        spk_id = []
        speaker_ids = self.speaker_ids.index(speaker_id)
        for i in range(num_utterances):
            # folder_path = os.path.join(self.file_path, speaker_id)
            rd_uttrance = np.random.choice(len(spk_utt), 1)[0]
            # utterance = self.utterance_ids[speaker_id][rd_uttrance]
            mel_spec = np.load(spk_utt[rd_uttrance])

            if mel_spec.shape[1] <= self.samples_length:
                while mel_spec.shape[1] < self.samples_length:

                    rd_uttrance = np.random.choice(len(spk_utt), 1)[0]
                    # utterance = self.utterance_ids[speaker_id][rd_uttrance]
                    mel_spec = np.load(spk_utt[rd_uttrance])

            rd_begin = np.random.choice((mel_spec.shape[1] - self.samples_length), 1)[0]
            mel_spec = mel_spec[:,rd_begin:rd_begin + self.samples_length]

            data.append(mel_spec)
            spk_id.append(speaker_ids)

        data = torch.tensor(data)
        spk_id = torch.tensor(spk_id)
        return data, spk_id

    def get_utterance(self, speaker, utterance):

        fp = os.path.join(self.file_path, speaker, utterance)
        mel_spec = np.load(fp)
        return mel_spec

    def __len__(self):
        return len(self.speaker_ids)
##############################################################################################
## the output of this dataset combines both mfcc and melspectrogram for training proposed AutoVC model
class SpeechDataset4(Dataset):
    def __init__(self, file_path,sr=16000, samples_length=64, num_utterances=10, male_dataset=False):
        self.file_path =file_path
        self.sr = sr
        self.samples_length = samples_length
        self.num_utterances = num_utterances
        self.speaker_info_fp = os.path.join(self.file_path, 'speaker-info.txt')
        self.utterance_ids = {}


        if male_dataset:
            self.speaker_ids = get_male_spk(self.file_path)
        else:
            self.speaker_ids = [f for f in os.listdir(self.file_path)]
        
        for speaker in self.speaker_ids:
            self.utterance_ids[speaker] = glob.glob(os.path.join(self.file_path, speaker, '*_mel.npy'))

    def __getitem__(self, index):
        speaker_id = self.speaker_ids[index]
        utterances = []
        data = []
        mfcc_data = []
        speaker_ids = []
        for i in range(self.num_utterances):
            folder_path = os.path.join(self.file_path, speaker_id)
            rd_uttrance = np.random.choice(len(self.utterance_ids[speaker_id]), 1)[0]
            utterance = self.utterance_ids[speaker_id][rd_uttrance]
            # mfcc = utterance.replace('npy', 'mfcc')
            mel_spec = np.load(utterance)
            mfcc = np.load(utterance.replace('mel', 'mfcc'))
            # print('load mel spec shape: ', mel_spec.shape)
            # transpose for new new_encoder2 dataset
            np.transpose(mel_spec, (-1, -2))
            if mel_spec.shape[1] <= self.samples_length:
                while mel_spec.shape[1] < self.samples_length:
                    rd_uttrance = np.random.choice(len(self.utterance_ids[speaker_id]), 1)[0]
                    utterance = self.utterance_ids[speaker_id][rd_uttrance]
                    mel_spec = np.load(utterance)

            rd_begin = np.random.choice((mel_spec.shape[1] - self.samples_length), 1)[0]
            mel_spec = mel_spec[:,rd_begin:rd_begin + self.samples_length]
            mfcc = mfcc[:,rd_begin:rd_begin + self.samples_length]
            # print('load mel spec shape: ', mel_spec.shape)
            # print('spk id: ', speaker_id)
            data.append(mel_spec)
            mfcc_data.append(mfcc)
            utterances.append(utterance)
            speaker_ids.append(speaker_id)
        
        
        data = torch.tensor(data)
        mfcc_data = torch.tensor(mfcc_data)
        # print('data shape: ', data.shape)
        return data, utterances, speaker_ids, mfcc_data


    def get_obama_samples(self):
        speaker_id = 'VCTK-Corpus_wav16_obama'
        utterances = []
        data = []
        speaker_ids = []
        for i in range(self.num_utterances):
            folder_path = os.path.join(self.file_path, speaker_id)
            rd_uttrance = np.random.choice(len(self.utterance_ids[speaker_id]), 1)[0]
            utterance = self.utterance_ids[speaker_id][rd_uttrance]
            mel_spec = np.load(utterance)

            if mel_spec.shape[1] <= self.samples_length:
                while mel_spec.shape[1] < self.samples_length:

                    rd_uttrance = np.random.choice(len(self.utterance_ids[speaker_id]), 1)[0]
                    utterance = self.utterance_ids[speaker_id][rd_uttrance]
                    mel_spec = np.load(utterance)

            rd_begin = np.random.choice((mel_spec.shape[1] - self.samples_length), 1)[0]
            mel_spec = mel_spec[:,rd_begin:rd_begin + self.samples_length]

            data.append(mel_spec)
            utterances.append(utterance)
            speaker_ids.append(speaker_id)

        data = torch.tensor(data)
         
        return data, utterances, speaker_ids

    def __len__(self):
        return len(self.speaker_ids)
###############################################################################################################
class SpeechDatasetFVAE(Dataset):
    def __init__(self, file_path,sr=16000, samples_length=64):
        self.file_path =file_path
        self.sr = sr
        self.samples_length = samples_length
        # self.num_utterances = num_utterances

        self.speaker_ids = [f for f in os.listdir(self.file_path)]
        self.utterance_fp = np.array([])
        for speaker in self.speaker_ids:
            # print(speaker)
            spk_utt = np.array(glob.glob(os.path.join(self.file_path,speaker, "*.npy")))
            self.utterance_fp = np.concatenate((self.utterance_fp, spk_utt), axis=None)

    def __getitem__(self, index):

        utterance = self.utterance_fp[index]
        utt2_idx = np.random.choice(len(self.utterance_fp), 1)[0]
        utterance2 = self.utterance_fp[utt2_idx]

        mel_spec = np.load(utterance)
        np.transpose(mel_spec, (-1, -2))
        mel_spec2 = np.load(utterance2)
        np.transpose(mel_spec2, (-1, -2))

        if mel_spec.shape[1] < self.samples_length:
            mel_spec = np.pad(mel_spec, ((0,0),(0,self.samples_length-mel_spec.shape[1])), 'constant', constant_values=0)

        else:
            rd_begin = np.random.choice((mel_spec.shape[1] - self.samples_length), 1)[0]
            mel_spec = mel_spec[:,rd_begin:rd_begin + self.samples_length]

        if mel_spec2.shape[1] < self.samples_length:
            mel_spec2 = np.pad(mel_spec2, ((0,0),(0,self.samples_length-mel_spec2.shape[1])), 'constant', constant_values=0)

        else:
            rd_begin = np.random.choice((mel_spec2.shape[1] - self.samples_length), 1)[0]
            mel_spec2 = mel_spec2[:,rd_begin:rd_begin + self.samples_length]


        return torch.from_numpy(mel_spec), torch.from_numpy(mel_spec2)

    def __len__(self):
        return len(self.utterance_fp)

    def get_batch_utterances(self, speaker_id, num_utterances):

        spk_utt = np.array(glob.glob(os.path.join(self.file_path, speaker_id, "*.npy")))
        rnd_idx = np.random.choice(len(spk_utt), num_utterances)
        utterances = []
        data = []
        speaker_ids = []
        spk_id = self.speaker_ids.index(speaker_id)
        for i in range(num_utterances):
            # folder_path = os.path.join(self.file_path, speaker_id)
            rd_uttrance = np.random.choice(len(spk_utt), 1)[0]
            # utterance = self.utterance_ids[speaker_id][rd_uttrance]
            mel_spec = np.load(spk_utt[rd_uttrance])

            if mel_spec.shape[1] <= self.samples_length:
                while mel_spec.shape[1] < self.samples_length:

                    rd_uttrance = np.random.choice(len(spk_utt), 1)[0]
                    # utterance = self.utterance_ids[speaker_id][rd_uttrance]
                    mel_spec = np.load(spk_utt[rd_uttrance])

            rd_begin = np.random.choice((mel_spec.shape[1] - self.samples_length), 1)[0]
            mel_spec = mel_spec[:,rd_begin:rd_begin + self.samples_length]

            data.append(mel_spec)
            # utterances.append(utterance)
            speaker_ids.append(spk_id)

        data = torch.tensor(data)
        speaker_ids = torch.tensor(speaker_ids)
        return data, speaker_ids

    def get_batch_speaker(self, utterance): 

        data = []
        speaker_id = []
        utt = []
        for spk in self.speaker_ids:
            mel_spec = np.load(os.path.join(self.file_path, spk, utterance))
            print('mel shape: ', mel_spec.shape[1])
            rd_begin = np.random.choice((mel_spec.shape[1] - self.samples_length), 1)[0]
            mel_spec = mel_spec[:,rd_begin:rd_begin + self.samples_length]

            data.append(mel_spec)
            speaker_id.append(spk)
            utt.append(utterance)

        data = torch.tensor(data)
        return data, utt, speaker_id
    
    def get_utterance(self, speaker, utterance):

        fp = os.path.join(self.file_path, speaker, utterance)
        mel_spec = np.load(fp)
        return mel_spec

####################################### Dataset using MCC feature ####################################################################
class SpeechDatasetMCC(Dataset):
    def __init__(self, file_path, sr=16000, samples_length=128):
        self.file_path =file_path
        self.sr = sr
        self.samples_length = samples_length
        # self.num_utterances = num_utterances

        self.speaker_ids = [f for f in os.listdir(self.file_path)]
        self.utterance_fp = np.array([])
        for speaker in self.speaker_ids:
            # print(speaker)
            spk_utt = np.array(glob.glob(os.path.join(self.file_path,speaker, "*.npz")))
            self.utterance_fp = np.concatenate((self.utterance_fp, spk_utt), axis=None)

    def __getitem__(self, index):

        utterance = self.utterance_fp[index]
        data = np.load(utterance)
        mc_norm, mc_mean, mc_std, f0 = data['normalized_mc'], data['mc_mean'], data['mc_std'], data['f0']

        mc_norm = mc_norm.T
        # print('mc_norm shape: ', mc_norm.shape)
        if mc_norm.shape[1] < self.samples_length:
            mc_norm = np.pad(mc_norm, ((0,0),(0,self.samples_length - mc_norm.shape[1])), 'constant', constant_values=0)
        else:
            rd_begin1 = np.random.choice((mc_norm.shape[1] - self.samples_length), 1)[0]
            mc_norm = mc_norm[:,rd_begin1:rd_begin1 + self.samples_length]

        utterance_id = utterance.split('/')[-1].split('.')[0]
        spk_id = utterance.split('/')[-2]
        spk_id = self.speaker_ids.index(spk_id)

        return torch.from_numpy(mc_norm), spk_id

    def __len__(self):
        return len(self.utterance_fp)

    def get_batch_utterances(self, speaker_id, num_utterances):

        spk_utt = np.array(glob.glob(os.path.join(self.file_path, speaker_id, "*.npy")))
        rnd_idx = np.random.choice(len(spk_utt), num_utterances)
        utterances = []
        data = []
        speaker_ids = []
        spk_id = self.speaker_ids.index(speaker_id)
        for i in range(num_utterances):
            # folder_path = os.path.join(self.file_path, speaker_id)
            rd_uttrance = np.random.choice(len(spk_utt), 1)[0]
            # utterance = self.utterance_ids[speaker_id][rd_uttrance]
            mel_spec = np.load(spk_utt[rd_uttrance])

            if mel_spec.shape[1] <= self.samples_length:
                while mel_spec.shape[1] < self.samples_length:

                    rd_uttrance = np.random.choice(len(spk_utt), 1)[0]
                    # utterance = self.utterance_ids[speaker_id][rd_uttrance]
                    mel_spec = np.load(spk_utt[rd_uttrance])

            rd_begin = np.random.choice((mel_spec.shape[1] - self.samples_length), 1)[0]
            mel_spec = mel_spec[:,rd_begin:rd_begin + self.samples_length]

            data.append(mel_spec)
            # utterances.append(utterance)
            speaker_ids.append(spk_id)

        data = torch.tensor(data)
        speaker_ids = torch.tensor(speaker_ids)
        return data, speaker_ids

    def get_batch_speaker(self, utterance): 

        data = []
        speaker_id = []
        utt = []
        for spk in self.speaker_ids:
            mel_spec = np.load(os.path.join(self.file_path, spk, utterance))
            print('mel shape: ', mel_spec.shape[1])
            rd_begin = np.random.choice((mel_spec.shape[1] - self.samples_length), 1)[0]
            mel_spec = mel_spec[:,rd_begin:rd_begin + self.samples_length]

            data.append(mel_spec)
            speaker_id.append(spk)
            utt.append(utterance)

        data = torch.tensor(data)
        return data, utt, speaker_id
    
    def get_utterance(self, speaker, utterance):

        fp = os.path.join(self.file_path, speaker, utterance)
        mel_spec = np.load(fp)
        return mel_spec

###############################################################################################################
class SpeechDatasetMCC2(Dataset):
    def __init__(self, file_path, sr=16000, samples_length=128):
        self.file_path =file_path
        self.sr = sr
        self.samples_length = samples_length
        # self.num_utterances = num_utterances

        self.speaker_ids = [f for f in os.listdir(self.file_path)]
        self.spk_utt = []
        self.utterance_fp = np.array([])
        for i in range(len(self.speaker_ids)):
            # print(speaker)
            spk_utt = np.array(glob.glob(os.path.join(self.file_path,self.speaker_ids[i], "*.npz")))
            np.random.shuffle(spk_utt)
            self.spk_utt.append(spk_utt)
            spk_utt1 = spk_utt[:spk_utt.shape[0]//2]
            spk_utt2 = spk_utt[spk_utt.shape[0]//2 : spk_utt.shape[0]//2 + spk_utt.shape[0]//2]
            spk_utt = [(spk_utt1[i],spk_utt2[i]) for i in range(len(spk_utt1))]
            spk_utt = np.array(spk_utt)
            # print('speaker utterances: ', spk_utt)
            if i == 0:
                self.utterance_fp = spk_utt
            else:
                self.utterance_fp = np.concatenate((self.utterance_fp, spk_utt), axis=0)
    
    def shuffle_data(self):
        self.utterance_fp = np.array([])

        for i in range(len(self.spk_utt)):
            utt = self.spk_utt[i] 
            np.random.shuffle(utt)
            utt1 = utt[:utt.shape[0]//2]
            utt2 = utt[utt.shape[0]//2 : utt.shape[0]//2 + utt.shape[0]//2]
            utt = [(utt1[i], utt2[i]) for i in range(utt1.shape[0])]
            utt = np.array(utt)
            if i == 0:
                self.utterance_fp = utt
            else:
                self.utterance_fp = np.concatenate((self.utterance_fp, utt), axis=0)
                
        np.random.shuffle(self.utterance_fp)

    def __getitem__(self, index):
        
        utterance1 = self.utterance_fp[index, 0]
        utterance2 = self.utterance_fp[index, 1]

        utt1 = np.load(utterance1)
        utt2 = np.load(utterance2)

        mc_norm1, mc_mean1, mc_std1, f01 = utt1['normalized_mc'], utt1['mc_mean'], utt1['mc_std'], utt1['f0']
        mc_norm2, mc_mean2, mc_std2, f02 = utt2['normalized_mc'], utt2['mc_mean'], utt2['mc_std'], utt2['f0']

        mc_norm1 = mc_norm1.T
        mc_norm2 = mc_norm2.T

        if mc_norm1.shape[1] <= self.samples_length:
            mc_norm1 = np.pad(mc_norm1, ((0,0),(0,self.samples_length - mc_norm1.shape[1])), 'constant', constant_values=0)
        else:
            rd_begin1 = np.random.choice((mc_norm1.shape[1] - self.samples_length), 1)[0]
            mc_norm1 = mc_norm1[:,rd_begin1:rd_begin1 + self.samples_length]
        if mc_norm2.shape[1] <= self.samples_length:
            mc_norm2 = np.pad(mc_norm2, ((0,0),(0,self.samples_length - mc_norm2.shape[1])), 'constant', constant_values=0)            
        else:
            rd_begin2 = np.random.choice((mc_norm2.shape[1] - self.samples_length), 1)[0]    
            mc_norm2 = mc_norm2[:,rd_begin2:rd_begin2 + self.samples_length]

        spk_id = utterance1.split('/')[-2]
        spk_id = self.speaker_ids.index(spk_id)

        return torch.from_numpy(mc_norm1), torch.from_numpy(mc_norm2), torch.tensor(spk_id)

    def __len__(self):
        return len(self.utterance_fp)

    def get_spk_utterances(self, speaker_id):

        spk_utt = np.array(glob.glob(os.path.join(self.file_path, speaker_id, "*.npz")))
        rnd_idx = np.random.choice(len(spk_utt), len(spk_utt))
        data = []
        aps = []
        f0s = []
        utt_ids = []
        for i in range(len(spk_utt)):
            
            utt_id = spk_utt[rnd_idx[i]].split('/')[-1].split('.')[0]
            utt = np.load(spk_utt[rnd_idx[i]])
            mc_norm, ap, f0 = utt['normalized_mc'], utt['ap'], utt['f0']
            # mc_norm = mc_norm.T

            # if mc_norm.shape[1] < self.samples_length:
            #     mc_norm = np.pad(mc_norm, ((0,0),(0,self.samples_length - mc_norm.shape[1])), 'constant', constant_values=0)
            # else:
            #     rd_begin1 = np.random.choice((mc_norm1.shape[1] - self.samples_length), 1)[0]
            #     mc_norm1 = mc_norm1[:,rd_begin1:rd_begin1 + self.samples_length]

            # rd_begin = np.random.choice((mel_spec.shape[1] - self.samples_length), 1)[0]
            # mel_spec = mel_spec[:,rd_begin:rd_begin + self.samples_length]

            data.append(mc_norm)
            aps.append(ap)
            f0s.append(f0)
            utt_ids.append(int(utt_id))
            # utterances.append(utterance)
            # speaker_ids.append(spk_id)

        data = torch.tensor(data)
        aps = torch.tensor(aps)
        f0s = torch.tensor(f0s)
        utt_ids = torch.tensor(utt_ids)
        # speaker_ids = torch.tensor(speaker_ids)

        return data, aps, f0s, utt_ids

    def get_batch_speaker(self, utterance): 

        data = []
        speaker_id = []
        utt = []
        for spk in self.speaker_ids:
            mel_spec = np.load(os.path.join(self.file_path, spk, utterance))
            print('mel shape: ', mel_spec.shape[1])
            rd_begin = np.random.choice((mel_spec.shape[1] - self.samples_length), 1)[0]
            mel_spec = mel_spec[:,rd_begin:rd_begin + self.samples_length]

            data.append(mel_spec)
            speaker_id.append(spk)
            utt.append(utterance)

        data = torch.tensor(data)
        return data, utt, speaker_id
    
    def get_utterance(self, speaker, utterance):

        fp = os.path.join(self.file_path, speaker, utterance)
        mel_spec = np.load(fp)
        return mel_spec

@timing
def load_batch(loader):
    return next(iter(loader))


def speaker_to_onehot(speaker_ids, speaker_all,num_classes=109, num_utterance=40):
    # onehot_embedding = torch.nn.functional.one_hot(torch.arange(0,109), num_classes=109)
    # speaker_onehot = [speaker for speaker in speaker_ids for i in range(num_utterance)]
    # speaker_ids = [spea for spea in speaker_ids]
    speaker_onehot = np.empty((len(speaker_ids)*num_utterance), dtype=np.int16)
    for j in range(len(speaker_ids)):
        for i in range(num_utterance):
            idx = speaker_all.index(speaker_ids[j])
            # print(idx)
            speaker_onehot[j*num_utterance+i] = idx

    return torch.tensor(speaker_onehot) 


def dump_wav2spectrogram():
    import pickle
    file_path = '/home/ubuntu/VCTK-Corpus/new_encoder3/'
    mel_file_path = os.path.join('/home/manhlt/extra_disk/VCTK-Corpus/mel_spectrogram')
    if not os.path.exists(mel_file_path):
        os.mkdir(mel_file_path)

    speaker_ids = [speaker for speaker in os.listdir(file_path)]
    for speaker in speaker_ids:
        os.mkdir(os.path.join(mel_file_path, speaker))

    for speaker in speaker_ids:
        speaker_fp = os.path.join(file_path, speaker)
        speaker_mel_sp = os.path.join(mel_file_path, speaker)
        utterances = [f for f in os.listdir(speaker_fp) if os.path.isfile(os.path.join(speaker_fp, f))]
        for utterance in utterances:
            wav, sr = librosa.load(os.path.join(speaker_fp, utterance), sr=16000)
            mel_spec = processing.melspectrogram(wav)
            file = open(os.path.join(speaker_mel_sp, utterance+'.pkl'),'wb')
            pickle.dump(mel_spec, file)
            print('Processing: ',utterance)
    # print(speaker_ids)

if __name__=='__main__':
    # from vocoder2waveform import build_model
    # from vocoder2waveform import wavegen
    import librosa
    import numpy as np
    import librosa.display
    import matplotlib.pyplot as plt

    torch.set_printoptions(precision=2)

    device = torch.device('cuda')
    file_path = '/home/ubuntu/vcc2016_WORLD_dataset/'
    # ckpt_path = '/home/manhlt/extra_disk/checkpoint_step001000000_ema.pth'

    # plt.figure()
    # data = np.zeros((80,67), dtype=np.float)
    # data[0:10] = 0.5
    # data[11:20] = 0.01
    
    dataset = SpeechDatasetMCC2(file_path, samples_length=64)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=4,
                            pin_memory=True, shuffle=True, drop_last=True)

    data1, data2, spk = next(iter(dataloader))

    print(spk)
    print('firs sample: ', data1.shape)
    mfccs = data1[0].cpu().detach().numpy()
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()

    plt.savefig('../mfccs.png')
    # print(data1.float())
    # print(dataset.utterance_fp)
    # 
    # print(data[0].shape)
    # print('this is mfcc: ', mfcc[0].shape)

    
    #### shape of utterances [utterances_id, speaker_ids] ############
    # batch = load_batch(dataloader)
    # print(data[0][9])
    # print(len(utterances))
    # print(len(speaker_ids))
    # print(batch[1][0][0])
    # print('-----------------------------------------------------------------------------------')
    # print(utterances)
    # print(utterances)
    # print(speaker_ids)
    # mel_spec = data[0][0].cpu().numpy()
    # librosa.display.specshow(mel_spec, x_axis='time', y_axis='mel', sr=16000)
    # plt.colorbar(format='%f')
    # plt.title('My life is fuck up')

    # plt.show()
    # mel_spec2 = mel_spec
    # mel_spec2[:,:20] = 2
    # plt.figure()
    # librosa.display.specshow(mel_spec2, x_axis='time', y_axis='mel', sr=16000, fmax=8000)
    # plt.show()
    # onehot = speaker_to_onehot(speaker_ids, dataset.speaker_ids)

    # dump_wav2spectrogram()

    
    #     librosa.output.write_wav(utterance[i][0]+'.wav', wav, sr=16000)
