# Many-to-Many Voice Conversion based on Variational Autoencoder

Code repository for paper [link](link)

Manh Luong, Viet Anh Tran under review in INTERSPEECH 2021

# Dataset:

We use VCTK-Corpus to train and estimate our proposed model, VCTK dataset can be found in this [link](https://datashare.is.ed.ac.uk/handle/10283/3443)

# Pretrained model:

pretrained model can be downloaded in this [link](https://drive.google.com/file/d/1TixHkqxPPRfxONraNJiTnZU9vcHwy9F4/view?usp=sharing)
Wavenet Vocoder: [link](https://drive.google.com/file/d/1Zksy0ndlDezo9wclQNZYkGi_6i7zi4nQ/view?usp=sharing)

# Requirements:

- Python 3.6 or newer.
- Pytorch 1.4 or newer.
- librosa.
- tensorboardX.
- wavenet_vocoder ``pip install wavenet_vocoder``

# Prepare data for training

1. Download and uncompress VCTK dataset.
2. Move extracted dataset in ``[home directory]``.
3. run command: ``export HOME=[home directory]``
4. run command: ``bash preprocessing.sh``.

# Usage

To train the model run the following command:
``bash training.sh``

To convert voice from source to target using pretrained model. Run the follwoing commands:

1. cd [Disentangled-VAE directory]
2. mkdir ./results/checkpoints
3. cp [your downloaded checkpoint] ./results/checkpoints/
4. Download pretrained model of Wavenet_vocoder
5. cp [downloaded Wavenet_Vocoder]/checkpoint_step001000000_ema.pth [Disentangled-VAE directory]
6. edit two variables: ``src_spk`` and ``trg_spk`` in file conversion.sh to your source and target speaker, respectively.
7. run command: ``bash conversion.sh``
