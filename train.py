import torch
import numpy as np 
# from model.variational_base_vae import ConvolutionalVSC
from model.disentangled_vae import ConvolutionalMulVAE
# from sparse_encoding.acvae import ConvolutionalGVAE
import argparse, os
from preprocessing.dataset import SpeechDatasetMCC2, SpeechDatasetGVAE
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from sparse_encoding.train_feature_selection import train_fs
import json

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2, metavar='BS',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--hidden-size', type=str, default='400', metavar='HS',
                        help='hidden sizes, separated by commas (default: 400)')
    parser.add_argument('--speaker_size', type=int, default=4, metavar='LS',
                        help='number of latent dimensions (default: 200)')
    parser.add_argument('--latent-size', type=int, default=32, metavar='LS',
                        help='number of latent dimensions (default: 200)')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', 
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=11, metavar='N',
                        help='number of epochs to train (default: 11)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--dataset', default='VCTK')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='LOG',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--report-interval', type=int, default=11, metavar='REP',
                        help='how many epochs to wait before storing training status')
    parser.add_argument('--sample-size', type=int, default=64, metavar='SS',
                        help='how many images to include in sample image')
    parser.add_argument('--do-not-resume', action='store_true', default=False,
                        help='retrains the model from scratch')
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='applies normalization')
    parser.add_argument('--beta_cof', default=0.1, type=float)
    parser.add_argument('--mse_cof', default=10, type=float)
    parser.add_argument('--kl_cof', default=10, type=float)
    parser.add_argument('--style_cof', default=0.1, type=float)
    parser.add_argument('--samples_length', default=128, type=int)
    return parser

def get_dataset(dataset_fp, batch_size, num_utt, shuffle_dataset=True):
    
    # sample length for past experiments is 64
    # dataset = SpeechDatasetMCC2(dataset_fp, samples_length=128)
    dataset = SpeechDatasetGVAE(dataset_fp, samples_length=64)

    train_loader = DataLoader(dataset, batch_size=batch_size,
                             pin_memory=True, shuffle=True)
    
    return train_loader, dataset



if __name__=='__main__':
    parse = get_parse()

    parse.add_argument('--alpha', default=0.01, type=float, metavar='A') # alpha = 0.5 achieve quite good results
    parse.add_argument('--dataset_fp', default='/root/VCTK-Corpus/Autovc-known-speakers', type=str)
    parse.add_argument('--log_dir', default='./results', type=str)
    parse.add_argument('--src_spk', default='VCTK-Corpus_wav16_p225', type=str)
    parse.add_argument('--trg_spk', default='VCTK-Corpus_wav16_p226', type=str)
    parse.add_argument('--train', type=bool, default=False)
    parse.add_argument('--convert', type=bool, default=False)
    # parse.add_argument('--gamma', default=6.4, type=float)
    args = parse.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    train_loader, dataset = get_dataset(args.dataset_fp, batch_size=args.batch_size, num_utt=40)

    if not os.path.exists('../' + args.log_dir):
        os.mkdir('../' + args.log_dir)

    config = vars(args)
    with open('../'+args.log_dir + '/config.json', 'w') as fp:
        json.dump(config, fp, indent=4)


    vsc = ConvolutionalMulVAE(args.dataset, 64, 80,
                          args.latent_size, args.lr,
                          args.alpha, args.log_interval, args.normalize, speaker_size=args.speaker_size,
                          latent_dim=args.latent_size, beta=args.beta_cof, batch_size=args.batch_size,
                          mse_cof=args.mse_cof, kl_cof=args.kl_cof, style_cof=args.style_cof)
    
    if args.train:
        vsc.run_training(train_loader, train_loader, args.epochs,
                    args.report_interval, args.sample_size, reload_model=True,
                    checkpoints_path=args.log_dir+'/checkpoints', images_path=args.log_dir+'/images',
                    logs_path=args.log_dir+'/logs', estimation_dir='../'+args.log_dir+'/images/estimation')

    # list_cv = [("VCTK-Corpus_wav16_p226","VCTK-Corpus_wav16_p225")]

    if args.convert:

        vsc.voice_conversion_mel(ckp_path=args.log_dir + '/checkpoints/',
                    generation_dir=args.log_dir+'/generation/',
                    src_spk=args.src_spk, trg_spk=args.trg_spk,
                    dataset_fp=args.dataset_fp)
