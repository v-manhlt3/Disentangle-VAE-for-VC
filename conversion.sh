# python train_acvae.py --dataset_fp=/home/ubuntu/vcc2018_WORLD_dataset --latent-size=32 --log_dir=VC_logs3/VCC2018_gvae_mcc_32_128_beta0.1
src_spk="VCTK-Corpus_wav16_p225"
trg_spk="VCTK-Corpus_wav16_p226"

python train.py --convert true --dataset_fp=$HOME/VCTK_mel \
                --latent-size=32  \
                --samples_length=128 \
                --batch-size=8 \
                --style_cof=10 \
                --mse_cof=10 \
                --style_cof=0.1 \
                --speaker_size=4 \
                --src_spk=$src_spk \
                --trg_spk=$trg_spk \
