import torch
import numpy as np 
import matplotlib.pyplot as plt
from sparse_encoding.feature_selection import FeatureSelection
from torch.optim import Adam
from tqdm import tqdm
import os
from glob import glob

def train_fs(vsc, dataloader, lr, save_path, init_epoch=0, load_model=False):

    if load_model:
        list_ckp = glob(os.path.join(save_path,'*.pth'))
        # print('list ckp: ', list_ckp)
        ckp_fp = list_ckp[-1]
        init_epoch = int(ckp_fp.split('/')[-1].split('_')[0])
        fs_model = FeatureSelection(512, 109).cuda()
        fs_model.load_state_dict(torch.load(ckp_fp))
    else:
        fs_model = FeatureSelection(512, 109).cuda()
    optimizer = Adam(fs_model.parameters(), lr=1e-3)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    vsc.model.eval()
    for epoch in range(init_epoch, 200000):
        total_loss = 0
        for iterations, (data,_,spk_ids) in enumerate(tqdm(dataloader)):
            data = data.cuda().float()
            spk_ids = torch.tensor(spk_ids).cuda()
            with torch.no_grad():
                mu, logvar, logspike = vsc.model.encode(data)
                latent_vectors = vsc.model.reparameterize(mu, logvar, logspike)
            weights = fs_model(latent_vectors)
            weighted_data = torch.mul(latent_vectors, weights)
            cls_spk = fs_model.classify(weighted_data)

            ce_loss = fs_model.calc_loss(weights, spk_ids, cls_spk)
            total_loss += ce_loss
            # loss = weight_loss + ce_loss
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()
            # print('Weight Loss: ', weight_loss.item())
            # print('Cross Entropy Loss: ', ce_loss.item())
        total_loss /= len(dataloader.dataset)
        test_batch,_,_ = next(iter(dataloader))
        test_batch = test_batch.cuda().float()
        with torch.no_grad():
                mu, logvar, logspike = vsc.model.encode(test_batch)
                latent_vectors = vsc.model.reparameterize(mu, logvar, logspike)
        maskes = fs_model(latent_vectors)
        print("Cross entropy Loss epoch:{} --- Loss:{}".format(epoch, total_loss))
        mask = [ele for ele in maskes[0] if ele>=0.7]
        zero_ele = [ele for ele in maskes[0] if ele <= 1e-3]
        print('The number of features: ', len(mask))
        print('The number of zeoros element: ', len(zero_ele))

        if epoch % 100 == 0:
            ckp_fp = os.path.join(save_path, str(epoch)+'_feature_selection.pth')
            torch.save(fs_model.state_dict(),ckp_fp)

def feature_selection(fs_model, latent_vectors):

    idx_vector = []

    with torch.no_grad():
        mask = fs_model(latent_vectors)
        
    # print('mask_idx: ', mask_idx)
    for i in range(mask.shape[0]):
        mask_idx = mask[i] == 1
        mask_idx = torch.nonzero(mask).cpu().detach().numpy()
        # print(' mask idx: ', mask_idx)
        if i == 0:
            idx_vector = mask_idx
        else:
            idx_vector = np.intersect1d(idx_vector, mask_idx)

    return idx_vector