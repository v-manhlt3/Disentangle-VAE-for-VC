# Many-to-Many Voice Conversion based on Variational Autoencoder

Code repository for paper [link](link)

Manh Luong, Viet Anh Tran under review in ICASSP 2021

# Dataset:

We use VCTK-Corpus to train and estimate our proposed model, VCTK dataset can be found in this [link](https://datashare.is.ed.ac.uk/handle/10283/3443)

# Pretrained model:

pretrained model can be downloaded in this [link](https://drive.google.com/file/d/1ScmvAWGk9mDTvkrp7pnRCd_YCS3JuDMU/view?usp=sharing)

# Requirements:

- Python 3.6 or newer
- Pytorch 1.4 or newer
- librosa
- tensorboardX

# Prepare data for training

1. Download and uncompress VCTK dataset.
2. Move extracted dataset in ``/home/ubuntu.
3. Go into ``preprocessing`` directory.
4. run command: ``python dataset_preprocess.py /home/ubuntu/ -o [output directory] -d VCTK --no_trim

# Usage

To train the model run the following command:
``python train.py --dataset_fp=[output directory] --latent-size=32 --log_dir=./log --epochs=2000 --report-interval=100 --lr=1e-4 --samples_length=64 --batch-size=16 --kl_cof=1.0 --mse_cof=1.0 --speaker_size=4 --train``

To convert voice from source to target using pretrained model. First, copy the pretrained model to folder ``./log/checkpoints`` and then run the following command:

``python train_acvae.py --dataset_fp=[output directory] --latent-size=32 --speaker_size=4 --log_dir=.log/``
