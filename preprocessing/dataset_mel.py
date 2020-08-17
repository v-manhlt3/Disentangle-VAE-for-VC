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


class SpeechDataset(Dataset):
    def __init__(self, file_path,sr=16000, samples_length=32768/2, num_utterances=10):
        self.file_path =file_path
        self.sr = sr
        self.samples_length = samples_length
        self.num_utterances = num_utterances

        self.speaker_ids = [f for f in os.listdir(self.file_path)]
        self.utterance_ids = {}
        # for speaker in self.speaker_ids:
        #     self.utterance_ids[speaker] = [f for f in os.listdir(os.path.join(self.file_path, speaker)) \
        #                                   if os.path.isfile(os.path.join(self.file_path, speaker, f))]
        for speaker in self.speaker_ids:
            self.utterance_ids[speaker] = glob.glob(os.path.join(self.file_path, speaker, '*.wav'))

    def __getitem__(self, index):
        speaker_id = self.speaker_ids[index]
        utterances = []
        data = []
        waveform = []

        speaker_ids = []
        for i in range(self.num_utterances):
            folder_path = os.path.join(self.file_path, speaker_id)
            rd_uttrance = np.random.choice(len(self.utterance_ids[speaker_id]), 1)[0]
            utterance = self.utterance_ids[speaker_id][rd_uttrance]
            fn = os.path.join(folder_path, utterance)
            wav, sr = librosa.load(fn, sr=self.sr)
            if len(wav) < self.samples_length:
                while len(wav) < self.samples_length:
                    # print('pick another wav')
                    rd_uttrance = np.random.choice(len(self.utterance_ids[speaker_id]), 1)[0]
                    utterance = self.utterance_ids[speaker_id][rd_uttrance]
                    fn = os.path.join(folder_path, utterance)
                    wav, sr = librosa.load(fn, sr=self.sr)
            rd_begin = np.random.choice((len(wav)-self.samples_length), 1)[0]
            wav = wav[rd_begin:rd_begin + self.samples_length]

            mel_spec = processing.melspectrogram(wav)[:,:64]

            data.append(mel_spec)
            utterances.append(utterance)
            waveform.append(wav)
            # speaker_ids.extend(speaker_id)
        data = torch.tensor(data)
        waveform = torch.tensor(waveform)

        return data, utterances, speaker_id, waveform
    def __len__(self):
        return len(self.speaker_ids)

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
        np.transpose(mel_spec, (-1, -2))

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
        spk_id = utterance.split('/')[-2].split('_')[-1]

        return torch.from_numpy(mel_spec), utterance_id, spk_id

    def __len__(self):
        return len(self.utterance_fp)

    def get_batch_utterances(self, speaker_id, num_utterances):

        spk_utt = np.array(glob.glob(os.path.join(self.file_path, speaker_id, "*.npy")))
        rnd_idx = np.random.choice(len(spk_utt), num_utterances)
        utterances = []
        data = []
        speaker_ids = []
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
            # speaker_ids.append(speaker_id)

        data = torch.tensor(data)
         
        return data

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
            speaker_ids.append(speaker_id)
        
        
        data = torch.tensor(data)
        # print('data shape: ', data.shape)
        return data, utterances, speaker_ids


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
 


##############################################################################################
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

    device = torch.device('cuda')
    file_path = '/home/ubuntu/vcc2016_train_librosa'
    # ckpt_path = '/home/manhlt/extra_disk/checkpoint_step001000000_ema.pth'

    # plt.figure()
    # data = np.zeros((80,67), dtype=np.float)
    # data[0:10] = 0.5
    # data[11:20] = 0.01
    
    dataset = SpeechDataset4(file_path, samples_length=64)
    # print(dataset.utterance_fp)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=4,
                            pin_memory=True, shuffle=True, drop_last=True)
    data, utt, spk_id, mfcc = next(iter(dataloader))
    print(data[0].shape)
    print('this is mfcc: ', mfcc[0].shape)

    
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
