# -*- coding: utf-8 -*-
import warnings
from data_process import data_lode
from data_process import load_sequences_features
from train import Train
from Utils.utils_ import *
from Config import Config
import dgl
from dgl import save_graphs
import pandas as pd
import os
warnings.filterwarnings("ignore")


def main_indep():
    config = Config()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    features, in_size = data_lode()
    smiles, fasta, mesh= load_sequences_features(config)
    device = config.device
    Hits_5 = torch.zeros(1, device=device)
    Hits_3 = torch.zeros(1, device=device)
    Hits_1 = torch.zeros(1, device=device)
    NDCG_5 = torch.zeros(1, device=device)
    NDCG_3 = torch.zeros(1, device=device)
    NDCG_1 = torch.zeros(1, device=device)
    MRR = torch.zeros(1, device=device)
    train_data_pos = np.array(
        pd.read_csv('./Data/indepent_data/train_data_pos.csv', header=None))
    train_data_neg = np.array(
        pd.read_csv('./Data/indepent_data/train_data_neg.csv',header=None))
    val_data_pos = np.array(
        pd.read_csv('./Data/indepent_data/test_data_pos.csv', header=None))
    val_data_neg = np.array(
        pd.read_csv('./Data/indepent_data/test_data_neg.csv', header=None))

    train_data_pos = torch.tensor(train_data_pos, device=device)
    train_data_neg = torch.tensor(train_data_neg, device=device)
    val_data_pos = torch.tensor(val_data_pos, device=device)
    val_data_neg = torch.tensor(val_data_neg, device=device)

    hg=construct_hg(train_data_pos,device=device)
    train_data = torch.cat((train_data_pos, train_data_neg), dim=0)
    val_data = torch.cat((val_data_pos, val_data_neg), dim=0)

    shuffle_index = torch.randperm(train_data.size(0))
    train_data = train_data[shuffle_index]

    result = Train(train_data, val_data, in_size, config, hg, features, smiles, fasta, mesh)
    Hits_5[0] = result[0]
    Hits_3[0] = result[1]
    Hits_1[0] = result[2]
    NDCG_5[0] = result[3]
    NDCG_3[0] = result[4]
    NDCG_1[0] = result[5]
    MRR[0] = result[6]
    print('----------independent test finished-----------')
    print('Independent test result：'
          'Hits@5:%.6f' % torch.mean(Hits_5).item(), 'Hits@3:%.6f' % torch.mean(Hits_3).item(), 'Hits@1:%.6f' % torch.mean(Hits_1).item(),
          'NDCG@5:%.6f' % torch.mean(NDCG_5).item(), 'NDCG@3:%.6f' % torch.mean(NDCG_3).item(), 'NDCG@1:%.6f' % torch.mean(NDCG_1).item(),
          'MRR:%.6f' % torch.mean(MRR).item())
    return torch.mean(Hits_5), torch.mean(Hits_3), torch.mean(Hits_1), torch.mean(NDCG_5), torch.mean(NDCG_3), torch.mean(NDCG_1), torch.mean(MRR)

def main_CV():
    config = Config()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    features, in_size = data_lode()
    smiles, fasta, mesh = load_sequences_features(config)
    device = config.device

    Hits_5, Hits_3, Hits_1 = [], [], []
    NDCG_5, NDCG_3, NDCG_1, MRR = [], [], [], []

    for fold_num in range(1, 6):
        train_data_pos = np.array(pd.read_csv(f'./Data/CV_data/CV_{fold_num}/train_data_pos.csv', header=None))
        train_data_neg = np.array(pd.read_csv(f'./Data/CV_data/CV_{fold_num}/train_data_neg.csv', header=None))
        val_data_pos = np.array(pd.read_csv(f'./Data/CV_data/CV_{fold_num}/val_data_pos.csv', header=None))
        val_data_neg = np.array(pd.read_csv(f'./Data/CV_data/CV_{fold_num}/val_data_neg.csv', header=None))

        train_data_pos = torch.tensor(train_data_pos, device=device)
        train_data_neg = torch.tensor(train_data_neg, device=device)
        val_data_pos = torch.tensor(val_data_pos, device=device)
        val_data_neg = torch.tensor(val_data_neg, device=device)

        hg = construct_hg(train_data_pos, device=device)

        train_data = torch.cat((train_data_pos, train_data_neg), dim=0)
        val_data = torch.cat((val_data_pos, val_data_neg), dim=0)

        shuffle_index = torch.randperm(train_data.size(0))
        train_data = train_data[shuffle_index]

        result = Train(train_data, val_data, in_size, config, hg, features, smiles, fasta, mesh)

        Hits_5.append(result[0])
        Hits_3.append(result[1])
        Hits_1.append(result[2])
        NDCG_5.append(result[3])
        NDCG_3.append(result[4])
        NDCG_1.append(result[5])
        MRR.append(result[6])

    Hits_5 = torch.tensor(Hits_5, device=device)
    Hits_3 = torch.tensor(Hits_3, device=device)
    Hits_1 = torch.tensor(Hits_1, device=device)
    NDCG_5 = torch.tensor(NDCG_5, device=device)
    NDCG_3 = torch.tensor(NDCG_3, device=device)
    NDCG_1 = torch.tensor(NDCG_1, device=device)
    MRR = torch.tensor(MRR, device=device)

    print('---------- 5-fold CV finished -----------')
    print('5-fold CV result：'
          'Hits@5:%.6f' % torch.mean(Hits_5).item(), 'Hits@3:%.6f' % torch.mean(Hits_3).item(), 'Hits@1:%.6f' % torch.mean(Hits_1).item(),
          'NDCG@5:%.6f' % torch.mean(NDCG_5).item(), 'NDCG@3:%.6f' % torch.mean(NDCG_3).item(), 'NDCG@1:%.6f' % torch.mean(NDCG_1).item(),
          'MRR:%.6f' % torch.mean(MRR).item())

    return torch.mean(Hits_5), torch.mean(Hits_3), torch.mean(Hits_1), torch.mean(NDCG_5), torch.mean(NDCG_3), torch.mean(NDCG_1), torch.mean(MRR)

if __name__ == '__main__':
    if not os.path.exists('./Result'):
        os.makedirs('./Result')
    config = Config()
    print('Starting the 5-fold CV experiment')
    CV_Hits5, CV_Hits3,CV_Hits1,CV_NDCG_5, CV_NDCG_3, CV_NDCG_1, CV_MRR_num=main_CV()
    with open('./Result/MTGNN_CV_print.txt', 'a') as f:
        f.write(f"{CV_Hits5.item():.6f}\t{CV_Hits3.item():.6f}\t{CV_Hits1.item():.6f}\t"
                f"{CV_NDCG_5.item():.6f}\t{CV_NDCG_3.item():.6f}\t{CV_NDCG_1.item():.6f}\t"
                f"{CV_MRR_num.item():.6f}\n")

    print('Starting the independent test experiment')
    indep_Hits5, indep_Hits3, indep_Hits1, indep_NDCG_5, indep_NDCG_3, indep_NDCG_1, indep_MRR_num=main_indep()
    with open('./Result/MTGNN_indep_print.txt', 'a') as f:
        f.write(f"{indep_Hits5.item():.6f}\t{indep_Hits3.item():.6f}\t{indep_Hits1.item():.6f}\t"
                f"{indep_NDCG_5.item():.6f}\t{indep_NDCG_3.item():.6f}\t{indep_NDCG_1.item():.6f}\t"
                f"{indep_MRR_num.item():.6f}\n")






