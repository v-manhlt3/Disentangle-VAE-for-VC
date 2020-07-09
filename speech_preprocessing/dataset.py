import os
import warnings

import torchaudio
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets.utils import download_url, extract_archive, walk_files
import numpy as np
from os import listdir
from os.path import isfile
import soundfile as sf

from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('Elapsed time: {}'.format(end-start))
        return result
    return wrapper

speaker_id =['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234',\
             'p236','p237','p238','p239','p240','p241','p243','p244','p245','p246',\
             'p247','p248','p249','p250','p251','p252','p253','p254','p255','p256',\
             'p257','p258','p259','p260','p261','p262','p263','p264','p265','p266',\
             'p267','p268','p269','p270','p271','p272','p273','p274','p275','p276',\
             'p277','p278','p279','p280','p281','p282','p283','p284','p285','p286',\
             'p287','p288','p292','p293','p294','p295','p297','p298','p299','p300',\
             'p301','p302','p303','p304','p305','p306','p307','p308','p310','p311',\
             'p312','p313','p314','p315','p316','p317','p318','p323','p326','p329',\
             'p330','p333','p334','p335','p336','p339','p340','p341','p343','p345',\
             'p347','p351','p360','p361','p362','p363','p364','p374','p376']
def load_audio_item(fileid, path, ext_audio,ext_txt, folder_audio, folder_txt, downsample=False):

    speaker_id, utterance_id = fileid.split("_")

    #Read text
    file_txt = os.path.join(path, folder_txt, speaker_id, speaker_id+"_"+utterance_id + ext_txt)
    with open(file_txt) as file_text:
        utterance = file_text.readlines()[0]

    #Read wav
    file_audio = os.path.join(path, folder_audio, speaker_id, fileid + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)
    if downsample:
        F = torchaudio.functional
        T = torchaudio.transform

        sample = T.Resample(sample_rate, 16000, resampling_method='sinc_interpolation')
        waveform = sample(waveform)
        waveform = F.dither(waveform, noise_shaping=True)
    
    return waveform, sample_rate, utterance, speaker_id, utterance_id

def read_folder(file_path):
    file_name = [f for f in listdir(file_path) if isfile(os.path.join(file_path,f))]
    print(file_name)
    return file_name

class Audio_Dataset(Dataset):

    _folder_txt="txt"
    _folder_audio="wav16"
    _ext_txt=".txt"
    _ext_audio=".wav"
    _except_folder="p315"

    def __init__(self,
                root="",
                url=None,
                folder_in_archive=None,
                download=False,
                downsample=False,
                transform=None,
                target_transform=None,
                dataset_path=""):
        if downsample:
            warnings.warn("downsample warning")
        if transform is not None or target_transform is not None:
            warnings.warn("transform warning, please remove the option transforms=True")

        self.downsample = downsample
        self.transform = transform
        self.target_transform = target_transform
        self._path = dataset_path
        # archive = os.path.basename(url)
        # self._path = os.path.join(root, folder_fo)
        walker = walk_files(
            self._path, suffix=self._ext_audio, prefix=False, remove_suffix=True
        )
        walker = filter(lambda w: self._except_folder not in w, walker)
        self._walker = list(walker)

    def __getitem__(self, n):

        fileid=self._walker[n]
        item = load_audio_item(
            fileid,
            self._path,
            self._ext_audio,
            self._ext_txt,
            self._folder_audio,
            self._folder_txt
        )

        waveform, sample_rate, utterance, speaker_id, utterance_id = item
        if self.transform is not None:
            waveform= self.transform(waveform)
        if self.target_transform is not None:
            utterance = self.target_transform(utterance)
        return waveform, sample_rate, utterance, speaker_id, utterance_id

    def __len__(self):
        return len(self._walker)


class Audio_Loader(object):
    def __init__(self, file_path, batch_size, num_utterance):
        self.batch_size = batch_size
        self.offset = 0
        self.speaker_id = [folder for folder in listdir(file_path)]
        self.rd_idx = np.random.choice(109, 109, replace=False)
        self.utterance_id = {}
        self.file_path = file_path
        self.num_utterance = num_utterance
        for i in range(len(speaker_id)):
            file_name = [f for f in listdir(os.path.join(file_path, speaker_id[i])) if isfile(os.path.join(file_path, speaker_id[i], f))]
            self.utterance_id[speaker_id[i]] = file_name
    @timing
    def get_batch(self):
        # print(self.utterance_id)
        batch_data = []
        batch_labels = []
        for i in range(self.batch_size):
            speaker_id = self.speaker_id[self.rd_idx[self.offset+i]]
            folder_path = os.path.join(self.file_path, speaker_id)
            utterance_arr = self.utterance_id[speaker_id]
            rd_utterance_idx = np.random.choice(len(utterance_arr), self.num_utterance)

            for j in range(self.num_utterance):
                file_name = os.path.join(folder_path, utterance_arr[rd_utterance_idx[j]])
                audio, sr = sf.read(file_name, dtype='float32')
                # print('length audio: ',len(audio))
                ## handle with some exception audio file
                if len(audio)-16000 < 0:
                    while len(audio) - 16000 < 0:
                        utterance_id_new = np.random.choice(len(utterance_arr), 1)[0]
                        file_name = os.path.join(folder_path, utterance_arr[utterance_id_new])
                        audio, sr = sf.read(file_name, dtype='float32')
                rd_begin = np.random.choice((len(audio)-16000),1)[0]
                # print('rd_idx: ',rd_begin)
                audio = audio[rd_begin:rd_begin+16000]
                # audio = torchvision.transforms.ToTensor(audio)
                batch_data.append(audio)
                batch_labels.append(self.speaker_id[i])

        self.offset+= self.batch_size
        batch_data = torch.tensor(batch_data)
        return batch_data, batch_labels

    def reset(self):
        self.offset=0
        self.rd_idx = np.random.choice(109, 109, replace=False)

if __name__=="__main__":
    file_path = "/home/manhlt/extra_disk/VCTK-Corpus/wav16"
    data_loader = Audio_Loader(file_path, 2, 5)
    data, labels = data_loader.get_batch()
    print(data.shape)
    print(data[0])
    print(labels)
    print(len(data_loader.utterance_id))
