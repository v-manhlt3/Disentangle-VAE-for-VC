import librosa
import numpy as np
import os
import pyworld
import pysptk
import time

def load_wavs(wav_dir, sr):
    
    debug_num = 0
    wavs = list()
    for file in os.listdir(wav_dir):
        if(file.count("wav") == 0):
            continue        
        """
        debug_num += 1
        if (debug_num > 100):
            break
        """
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr = sr, mono = True)
        wav = librosa.util.normalize(wav, norm=np.inf, axis=None)
        #wav = wav.astype(np.float64)
        wavs.append(wav)

    return wavs


def world_decompose(wav, fs, frame_period = 5.0, num_mcep=36):

    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)
    alpha = pysptk.util.mcepalpha(fs)
    mc = pysptk.sp2mc(sp, order=num_mcep-1, alpha=alpha)

    return f0, timeaxis, sp, ap, mc

def world_encode_spectral_envelop(sp, fs, dim = 24):

    # Get Mel-cepstral coefficients (MCEPs)

    #sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)

    return coded_sp

def world_decode_mc(mc, fs):

    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    #coded_sp = coded_sp.astype(np.float32)
    #coded_sp = np.ascontiguousarray(coded_sp)
    alpha = pysptk.util.mcepalpha(fs)
    sp = pysptk.mc2sp(mc, alpha, fftlen)
    # decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)

    return sp

def world_encode_data(wav, fs, frame_period = 5.0, num_mcep = 36):

    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    mcs = list()

    # for wav in wavs:
    f0, timeaxis, sp, ap, mc = world_decompose(wav = wav, fs = fs, frame_period = frame_period, num_mcep=num_mcep)
        # f0s.append(f0)
        # timeaxes.append(timeaxis)
        # sps.append(sp)
        # aps.append(ap)
        # mcs.append(mc)

    return f0, timeaxis, sp, ap, mc


def transpose_in_list(lst):

    transposed_lst = list()
    for array in lst:
        transposed_lst.append(array.T)
    return transposed_lst


def world_decode_data(coded_sps, fs):

    decoded_sps =  list()

    for coded_sp in coded_sps:
        decoded_sp = world_decode_spectral_envelop(coded_sp, fs)
        decoded_sps.append(decoded_sp)

    return decoded_sps


def world_speech_synthesis(f0, sp, ap, fs, frame_period):

    #decoded_sp = decoded_sp.astype(np.float64)
    wav = pyworld.synthesize(f0, sp, ap, fs, frame_period)
    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float32)

    return wav


def world_synthesis_data(f0s, decoded_sps, aps, fs, frame_period):

    wavs = list()

    for f0, decoded_sp, ap in zip(f0s, decoded_sps, aps):
        wav = world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period)
        wavs.append(wav)

    return wavs


def mcs_normalization_fit_transoform(mcs):

    # mcs_concatenated = np.concatenate(mcs, axis = 1)
    mcs_concatenated = mcs
    mcs_mean = np.mean(mcs_concatenated, axis = 1, keepdims = True)
    mcs_std = np.std(mcs_concatenated, axis = 1, keepdims = True)

    mcs_normalized = list()
    for mc in mcs:
        mcs_normalized.append((mc - mcs_mean) / mcs_std)
    
    return mcs_normalized, mcs_mean, mcs_std

def coded_sps_normalization_transoform(coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)
    
    return coded_sps_normalized

def coded_sps_normalization_inverse_transoform(normalized_coded_sps, coded_sps_mean, coded_sps_std):

    coded_sps = list()
    for normalized_coded_sp in normalized_coded_sps:
        coded_sps.append(normalized_coded_sp * coded_sps_std + coded_sps_mean)

    return coded_sps

def coded_sp_padding(coded_sp, multiple = 4):

    num_features = coded_sp.shape[0]
    num_frames = coded_sp.shape[1]
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    coded_sp_padded = np.pad(coded_sp, ((0, 0), (num_pad_left, num_pad_right)), 'constant', constant_values = 0)

    return coded_sp_padded

def wav_padding(wav, sr, frame_period, multiple = 4):

    assert wav.ndim == 1 
    num_frames = len(wav)
    num_frames_padded = int((np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values = 0)

    nlen = len(wav_padded) + 80
    a = 2**5
    wav_padded = np.pad(wav_padded, (0, (a-(nlen//80)%a)*80 - (nlen%80)), 'constant', constant_values = 0)
    
    return wav_padded


def logf0_statistics(f0s):

    # log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_concatenated = np.ma.log(f0s)
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std

def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):

    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    return f0_converted

def wavs_to_specs(wavs, n_fft = 1024, hop_length = None):

    stfts = list()
    for wav in wavs:
        stft = librosa.stft(wav, n_fft = n_fft, hop_length = hop_length)
        stfts.append(stft)

    return stfts


def wavs_to_mfccs(wavs, sr, n_fft = 1024, hop_length = None, n_mels = 128, n_mfcc = 24):

    mfccs = list()
    for wav in wavs:
        mfcc = librosa.feature.mfcc(y = wav, sr = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels, n_mfcc = n_mfcc)
        mfccs.append(mfcc)

    return mfccs


def mfccs_normalization(mfccs):

    # mfccs_concatenated = np.concatenate(mfccs, axis = 1)
    mfccs_concatenated = mfccs
    mfccs_mean = np.mean(mfccs_concatenated, axis = 1, keepdims = True)
    mfccs_std = np.std(mfccs_concatenated, axis = 1, keepdims = True)
    mfccs_normalized = (mfccs -mfccs_mean) /  mfccs_std
    # mfccs_normalized = list()
    # for mfcc in mfccs:
    #     mfccs_normalized.append((mfcc - mfccs_mean) / mfccs_std)
    
    return mfccs_normalized, mfccs_mean, mfccs_std


def sample_train_data(dataset_A, dataset_B, n_frames = 128):

    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()

    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        assert frames_A_total >= n_frames
        start_A = np.random.randint(frames_A_total - n_frames + 1)
        end_A = start_A + n_frames
        train_data_A.append(data_A[:,start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]
        assert frames_B_total >= n_frames
        start_B = np.random.randint(frames_B_total - n_frames + 1)
        end_B = start_B + n_frames
        train_data_B.append(data_B[:,start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)

    return train_data_A, train_data_B


def preprocess_voice(data_dir, name, sampling_rate = 16000, num_mcep = 36, frame_period = 5.0, n_frames = 1024):

    print('Preprocessing Data...')

    start_time = time.time()

    print("Data Loading...")

    wavs = load_wavs(wav_dir = data_dir, sr = sampling_rate)

    print("Extracting f0 and mcep...")
    
    f0s, timeaxes, sps, aps, mcs = world_encode_data(wavs = wavs, fs = sampling_rate, frame_period = frame_period, num_mcep=num_mcep)

    del wavs, timeaxes, sps, aps

    log_f0s_mean, log_f0s_std = logf0_statistics(f0s)

    print("Saving f0 Data...")
    np.savez(os.path.join(data_dir, "log_f0_" + name + '.npz'), mean = log_f0s_mean, std = log_f0s_std)

    del f0s, log_f0s_mean, log_f0s_std

    mcs_t = transpose_in_list(lst = mcs)

    del mcs

    mcs_norm, mcs_mean, mcs_std = mcs_normalization_fit_transoform(mcs = mcs_t)

    print("Saving mcep Data...")
    np.savez(os.path.join(data_dir, "mcep_" + name + '.npz'), mean = mcs_mean, std = mcs_std)    

    end_time = time.time()
    time_elapsed = end_time - start_time

    print('Preprocessing Done.')

    print('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, (time_elapsed % 3600 // 60), (time_elapsed % 60 // 1)))


if __name__=='__main__':
    import argparse, os
    from glob import glob
    from tqdm import tqdm
    import librosa.display
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_fp', default='/home/ubuntu/vcc2018_training/', type=str)
    parser.add_argument('--out_dir', default='/home/ubuntu/vcc2018_WORLD_dataset_norm', type=str)
    parser.add_argument('--stat_fp', default='/home/ubuntu/vcc2018_stat_data/', type=str)
    args = parser.parse_args()
    # SR = 22050
    SR =16000
    dataset_fp = args.dataset_fp

    speaker_ids = [spk for spk in os.listdir(dataset_fp)]
    # for spk in speaker_ids:
    #     if not os.path.exists(os.path.join(args.out_dir, spk)):
    #         os.mkdir(os.path.join(args.out_dir, spk))
    #     utterances =  glob(os.path.join(dataset_fp, spk, "*.wav"))
    #     logf0 = np.load(args.stat_fp + 'log_f0_' + str(spk) + '.npz')
    #     mcep = np.load(args.stat_fp + 'mcep_' + str(spk) + '.npz')
    #     print('preprocessing speaker: ', spk)
    #     for utt in tqdm(utterances):
            
    #         output_fn = utt.replace('wav', 'npz').split('/')[-1]
    #         output_fp = os.path.join(args.out_dir, spk, output_fn)
    #         wav, sr = librosa.load(utt, sr=SR)
    #         wav = librosa.util.normalize(wav)

            

    #         f0, timeaxis, sp, ap, mcc = world_encode_data(wav=wav, fs=SR)
    #         mcc_transposed = np.array(mcc).T
    #         # print('mcc shape: ', mcc.shape)
            
    #         # mcc = mcc

    #         mc_mean = mcep['mean']
    #         # print('mcc mean: ', mc_mean.shape)
    #         # mc_mean = np.repeat(mc_mean, mcc.shape[0], axis=1)
    #         mc_std = mcep['std']
    #         # mc_std = np.repeat(mc_std, mcc.shape[0], axis=1)

    #         normalized_mc = (mcc_transposed - mc_mean) /  mc_std

    #         np.savez(output_fp, f0=f0, sp=sp, ap=ap, mcc=mcc, normalized_mc=normalized_mc, mc_mean=mcep['mean'].T, mc_std=mcep['std'].T)


    # fp = '/home/ubuntu/vcc2018_WORLD_dataset_norm/VCC2TM2/10001.npz'
    # data = np.load(fp)
    # mcc_norm = data['normalized_mc']
    # mcc = data['mcc']

    # plt.figure()
    # librosa.display.specshow(mcc_norm, x_axis='time', y_axis='mel', sr=16000)
    # plt.colorbar()
    # plt.title('origin_MFCC')
    # plt.tight_layout()

    # plt.savefig('/home/ubuntu/mcc_normalization1.png')


    # plt.figure()
    # librosa.display.specshow(mcc, x_axis='time', y_axis='mel', sr=16000)
    # plt.colorbar()
    # plt.title('MFCC')
    # plt.tight_layout()

    # plt.savefig('/home/ubuntu/mfccs1.png')