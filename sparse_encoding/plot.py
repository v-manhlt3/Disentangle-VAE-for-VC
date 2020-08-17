import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import os

## function is used to plot the sparse latent code
def encoding_visualization(z, estimation_dir, i, epoch):

    latent_vector = z.cpu().detach().numpy()
    index = np.arange(0,latent_vector.shape[0],1)
    # value =  np.random.uniform(0, 5, 512)

    # Draw plot
    plt.figure()
    fig, ax = plt.subplots(figsize=(16,10), dpi= 80)
    ax.vlines(x=index, ymin=0, ymax=latent_vector, color='firebrick', alpha=0.7, linewidth=2)
    ax.scatter(x=index, y=latent_vector, s=75, color='firebrick', alpha=0.7)
    plt.savefig(os.path.join(estimation_dir, str(epoch)+'_latent_code_'+str(i)+'.png'))


## function is used to plot the analysis of latent code of a particualar speaker
def plot_latentvt_analysis(latent_vectors, estimation_dir, speaker_id,
                           threshold_mean=0.1, threshold_std=0.2):

    latent_vectors = latent_vectors.cpu().detach().numpy()
    # print('latent vectors: ', latent_vectors.shape)
    # print(latent_vectors[0])
    mean = np.mean(latent_vectors, axis=0)
    std = np.std(latent_vectors, axis=0)
    idx = np.arange(mean.shape[0])
    

    threshold_mean_idx = np.where(np.abs(mean) < threshold_mean)[0]
    threshold_std_idx = np.where(std > threshold_std)[0]
    threshold_idx = np.concatenate([threshold_mean_idx, threshold_std_idx])
    threshold_idx = np.unique(threshold_idx)

    idx = np.delete(idx, threshold_idx, axis=0)
    print('mean shape: ', mean.shape)
    print('std shape: ', std.shape)
    mean_vector = np.savez(os.path.join(estimation_dir, speaker_id+'_analysis'), mean=mean, std=std, index=idx)

    mean = np.delete(mean, threshold_idx, axis=0)
    std = np.delete(std, threshold_idx, axis=0)
    

    # mean_vector = np.savez(os.path.join(estimation_dir, speaker_id+'_analysis'), mean=mean, std=std, index=idx)

    plt.figure(figsize=(15,5))
    plt.xticks(idx)
    plt.errorbar(idx, mean, std, linestyle='None', marker='o')
    plt.savefig(os.path.join(estimation_dir, speaker_id+'_latentcode_analysis.png'))
    
    return idx, mean, std


    

