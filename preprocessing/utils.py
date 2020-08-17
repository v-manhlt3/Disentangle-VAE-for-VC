import librosa
import librosa.filters
import numpy as np
from preprocessing.hparams import hparams
# from hparams import hparams
# from hparams_autovc import hparams
from scipy.io import wavfile
import math
import pyworld as pw
import torch




def load_wav(path):
    return librosa.core.load(path, sr=hparams.sample_rate)[0]


def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, hparams.sample_rate, wav.astype(np.int16))


def trim(quantized):
    start, end = start_and_end_indices(quantized, hparams.silence_threshold)
    return quantized[start:end]


def adjust_time_resolution(quantized, mel):
    """Adjust time resolution by repeating features
    Args:
        quantized (ndarray): (T,)
        mel (ndarray): (N, D)
    Returns:
        tuple: Tuple of (T,) and (T, D)
    """
    assert len(quantized.shape) == 1
    assert len(mel.shape) == 2

    upsample_factor = quantized.size // mel.shape[0]
    mel = np.repeat(mel, upsample_factor, axis=0)
    n_pad = quantized.size - mel.shape[0]
    if n_pad != 0:
        assert n_pad > 0
        mel = np.pad(mel, [(0, n_pad), (0, 0)], mode="constant", constant_values=0)

    # trim
    start, end = start_and_end_indices(quantized, hparams.silence_threshold)

    return quantized[start:end], mel[start:end, :]
adjast_time_resolution = adjust_time_resolution  # 'adjust' is correct spelling, this is for compatibility


def start_and_end_indices(quantized, silence_threshold=2):
    for start in range(quantized.size):
        if abs(quantized[start] - 127) > silence_threshold:
            break
    for end in range(quantized.size - 1, 1, -1):
        if abs(quantized[end] - 127) > silence_threshold:
            break

    assert abs(quantized[start] - 127) > silence_threshold
    assert abs(quantized[end] - 127) > silence_threshold

    return start, end


def melspectrogram(y):
    D = _lws_processor().stft(y).T
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hparams.ref_level_db
    if not hparams.allow_clipping_in_normalization:
        assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    return _normalize(S)


def get_hop_size():
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size


def _lws_processor():
    import lws
    return lws.lws(hparams.fft_size, get_hop_size(), mode="speech")


def lws_num_frames(length, fsize, fshift):
    """Compute number of time frames of lws spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def lws_pad_lr(x, fsize, fshift):
    """Compute left and right padding lws internally uses
    """
    M = lws_num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r

# Conversions:


_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, hparams.fft_size,
                               fmin=hparams.fmin, fmax=hparams.fmax,
                               n_mels=hparams.num_mels)


def _amp_to_db(x):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db

# print(hparams.fft_size)
######################### Operation for pitch tracking ##################################

pitch_bins = 257

def pitch_tracking(wav, mel):
    result = np.zeros((1,mel.shape[1]))
    pitch, mag = librosa.piptrack(y=wav, sr=16000, S=mel)
    for i in range(pitch.shape[1]):
        result[0][i] = np.max(pitch[:,i])

    mean = get_mean(result)
    var = get_variance(result)
    result = (result-mean)/var

    return result

def get_mean(arr):
    mean = 0
    # mean = np.mean(arr)
    for ele in arr:
        mean += ele
    return mean/(arr.shape[0])

def get_variance(arr, mean):
    variance = 0
    for ele in arr:
        variance += math.pow((ele-mean), 2)
    # var = np.var(arr, axis=0)
    return variance/(arr.shape[0])

# def discrete_pitch(pitch_frame):

#     for idx in range(pitch_frame.shape[1]):
#         if pitch_frame[0][idx] < -0.9:
#             pitch_frame[0][idx] = 0
#         else

def estimate_pitch(segment, sr, fmin=50.0, fmax=2000.0):
    
    f_0, t = pw.dio(segment, sr)
    f_0_min = np.min(f_0)
    f_0_max = np.max(f_0)

    f_0 = (f_0 - f_0_min) / f_0_max
    f_0 = np.ceil(f_0*256)

    return f_0

def get_batch_pitch(batch_data, sr):

    batch_pitch = []
    for data in batch_data:
        # print('waveform data: ', data)
        data = data.detach().cpu().numpy().astype(np.float64)
        pitch = estimate_pitch(data, sr=sr)
        # print('pitch shape: ', pitch.shape)
        batch_pitch.append(pitch)
    # print('batch pitch data: ', batch_pitch)
    return torch.from_numpy(np.stack(batch_pitch)) 
    