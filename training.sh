
python train.py --train true \
                --dataset_fp=$HOME/VCTK_mel \
                --latent-size=32 \
                --epochs=200000 \
                --report-interval=500 \
                --lr=1e-4 \
                --samples_length=64 \
                --batch-size=8 \
                --style_cof=10 \
                --mse_cof=10 \
                --style_cof=0.1 \
                --speaker_size=4 \
