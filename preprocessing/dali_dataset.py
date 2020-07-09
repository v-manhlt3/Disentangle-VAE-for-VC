import torch 
from torch.utils.data import DataLoader, Dataset

from functools import wraps
from time import time

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali as dali
import numpy as np
import librosa
import utils


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

class SpectrogramPipeline(Pipeline):
    def __init__(self, device, batch_size, file_path):
        super(SpectrogramPipeline, self).__init__(batch_size, num_threads=1, device_id=0)

        self.device = device
        self.file_path = file_path
        self.speaker_ids = [f for f in os.listdir(self.file_path)]
        self.utterance_ids = {}
        for speaker in self.speaker_ids:
            self.utterance_ids[speaker] = [f for f in os.listdir(os.path.join(self.file_path, speaker)) \
                                          if os.path.isfile(os.path.join(self.file_path, speaker, f))]
        
        self.rd_idx = np.random.choice(len(self.speaker_ids), len(self.speaker_ids), replace=True)

        self.offset = 0
        self.batch_data = []
        for i in range(self.batch_size):
            speaker = self.speaker_ids[self.offset+i]
            rd_utterance = np.random.choice(len(self.utterance_ids), 1)[0]
            utterance = self.utterance_ids[speaker][rd_utterance]
            fn = os.path.join(self.file_path, speaker, utterance)
            wav, sr = librosa.load(fn, sr=16000)
            if len(wav) < 32768:
                while len(wav) < 32768:
                    rd_utterance = np.random.choice(len(self.utterance_ids), 1)[0]
                    utterance = self.utterance_ids[speaker][rd_utterance]
                    fn = os.path.join(self.file_path, speaker, utterance)
                    wav, sr = librosa.load(fn, sr=16000)
            rd_begin = np.random.choice((len(wav) - 32768),1)[0]
            wav = wav[rd_begin:rd_begin+32768]
            mel_spec = utils.melspectrogram(wav)
            self.batch_data.append(mel_spec)
        self.batch_data = torch.tensor(self.batch_data)
        # self.offset +=self.batch_size
        # self.

