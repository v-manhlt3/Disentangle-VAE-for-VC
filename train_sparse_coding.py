import torch
import numpy as np 
from sparse_encoding.conv_vae import ConvolutionalVSC
from sparse_encoding.conv_vae2 import ConvolutionalACVAE
import argparse, os
from preprocessing.dataset_mel import SpeechDataset3, SpeechDataset2
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='input batch size for training (default: 32)')
    # Hidden size for CelebA: 2000 dimensions, 2 layers
    parser.add_argument('--hidden-size', type=str, default='400', metavar='HS',
                        help='hidden sizes, separated by commas (default: 400)')
    # Latent size for CelebA: 800 dimensions
    parser.add_argument('--latent-size', type=int, default=256, metavar='LS',
                        help='number of latent dimensions (default: 200)')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', 
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=11, metavar='N',
                        help='number of epochs to train (default: 11)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--dataset', default='mnist',
                        help='dataset [mnist, fashion, celeba]')
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
    return parser

def get_dataset(dataset_fp, batch_size, num_utt, shuffle_dataset=True):
    # dataset = SpeechDataset3(dataset_fp, samples_length=64,
    #                         num_utterances=num_utt, male_dataset=False)
    dataset = SpeechDataset2(dataset_fp, samples_length=64)
    train_size = int(0.95 * len(dataset))
    test_size = len(dataset) - train_size

    print('Training dataset size: ', train_size)
    print('Testing dataset size: ', test_size)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             pin_memory=True, shuffle=True)
    
    return train_loader, test_loader, dataset


if __name__=='__main__':
    parse = get_parse()

    parse.add_argument('--alpha', default=0.01, type=float, metavar='A') # alpha = 0.5 achieve quite good results
    parse.add_argument('--dataset_fp', default='/home/ubuntu/vcc2016_train', type=str)
    parse.add_argument('--log_dir', default='results', type=str)
    args = parse.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader, dataset = get_dataset(args.dataset_fp, batch_size=args.batch_size, num_utt=10)

    print('train loader size: ', len(train_loader))
    print('test loader size: ', len(test_loader))

    vsc = ConvolutionalVSC(args.dataset, 64, 80,
                          args.latent_size, args.lr,
                          args.alpha, args.log_interval, args.normalize,
                          latent_dim=args.latent_size, beta=args.beta_cof, batch_size=args.batch_size)

    vsc.run_training(train_loader, test_loader, args.epochs,
                    args.report_interval, args.sample_size, reload_model=True,
                    checkpoints_path='../'+args.log_dir+'/checkpoints', images_path='../'+args.log_dir+'/images',
                    logs_path='../'+args.log_dir+'/logs', estimation_dir='../'+args.log_dir+'/images/estimation')

    # vsc.estimate_trained_model(test_loader)
    # vsc.generate_wav(test_loader, ckp_path='../'+args.log_dir+'/checkpoints',
    #                 generation_dir='../'+args.log_dir+'/generation/')

    # vsc.analyze_latent_code(speaker_id='vcc2016_training_SF2', ckp_path='../'+args.log_dir+'/checkpoints',
    #                         estimation_dir='../'+args.log_dir+'/generation', dataset=dataset)

    # vsc.voice_conversion(target_speaker='vcc2016_training_SM1', source_speaker='vcc2016_training_SF2',
    #                     utterance_id='100001.npy', dataset=dataset, ckp_path='../'+args.log_dir+'/checkpoints',
    #                     generation_dir='../'+args.log_dir+'/generation')