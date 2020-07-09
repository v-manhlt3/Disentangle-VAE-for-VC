import torch
from torch.utils.data import DataLoader, Dataset
from preprocessing.encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from pathlib import Path

import librosa
import librosa.display


clean_data_root = '/home/ubuntu/LibriSpeech/mel_spectrogram/'
filename2 = '/home/ubuntu/testtt.png'
clean_data_root = Path(clean_data_root)
dataset = SpeakerVerificationDataset(clean_data_root)
loader = SpeakerVerificationDataLoader(
        dataset,
        64,
        10,
        num_workers=8,)

speaker_batch = next(iter(loader))
print(speaker_batch.data.shape)
data = speaker_batch.data[0]
plt.figure()
plt.title('original mel spectrogram')
librosa.display.specshow(data, x_axis='time', y_axis='mel', sr=16000)
plt.colorbar(format='%f')
plt.savefig(filename2)