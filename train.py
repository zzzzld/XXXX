# -*- coding: utf-8 -*-
from model import MTGNN
from Utils.utils_ import *
import warnings
import torch


warnings.filterwarnings("ignore")


def Train(train_data, test_data, in_size, config, hg, features, smiles, fasta, mesh):
    np.random.seed(config.seed)

    val_data_pos = test_data[test_data[:, -1] == 1]

    shuffle_index = torch.randperm(test_data.size(0))
    task_test_data = test_data[shuffle_index]

    device = config.device

    model = MTGNN(
        meta_paths=config.metapaths,
        test_data=val_data_pos,
        in_size=in_size,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        dropout=config.dropout,
        etypes=config.etypes,
        config=config,
        device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)

    myloss = Myloss(device)
    mrr = MRR().to(device)
    matrix = Matrix().to(device)
    trainloss = []
    valloss = []
    result_list = []
    hits_max_matrix = torch.zeros((1, 3), device=device)
    NDCG_max_matrix = torch.zeros((1, 3), device=device)
    patience_num_matrix = torch.zeros((1, 1), device=device)
    MRR_max_matrix = torch.zeros((1, 1), device=device)
    epoch_max_matrix = torch.zeros((1, 1), device=device)

    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()

        score_train_predict = model(hg, features, train_data, smiles, fasta, mesh)
        train_label = torch.unsqueeze(train_data[:, 3], 1).to(device=device)
        train_loss = myloss(score_train_predict, train_label, config.loss_gamma)
        trainloss.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            score_val_predict = model(hg, features, task_test_data, smiles, fasta, mesh)
            val_label = task_test_data[:, 3].unsqueeze(1).to(device=device, dtype=torch.float)
            val_loss = myloss(score_val_predict, val_label, config.loss_gamma)
            valloss.append(val_loss.item())
            predict_val = score_val_predict.detach()

            hits5, ndcg5, sample_hit5, sample_ndcg5 = matrix(5, 10, predict_val, len(val_data_pos), shuffle_index)
            hits3, ndcg3, sample_hit3, sample_ndcg3 = matrix(3, 10, predict_val, len(val_data_pos), shuffle_index)
            hits1, ndcg1, sample_hit1, sample_ndcg1 = matrix(1, 10, predict_val, len(val_data_pos), shuffle_index)
            MRR_num, sample_mrr = mrr(10, predict_val, len(val_data_pos), shuffle_index)
            result = [val_loss.item()] + [hits5] + [hits3] + [hits1] + [ndcg5] + [ndcg3] + [ndcg1] + [MRR_num]
            result_list.append(result)
            if (epoch + 1) % 10 == 0:
                print('Epoch:', epoch + 1, 'Train loss:%.4f' % train_loss.item(),
                      'Val Loss:%.4f' % result_list[epoch][0], 'Hits@5:%.6f' % result_list[epoch][-7],
                      'Hits@3:%.6f' % result_list[epoch][-6], 'Hits@1:%.6f' % result_list[epoch][-5],
                      'NDCG@5:%.6f' % result_list[epoch][-4], 'NDCG@3:%.6f' % result_list[epoch][-3],
                      'NDCG@1:%.6f' % result_list[epoch][-2], 'MRR:%.6f' % result_list[epoch][-1])
            patience_num_matrix = ealy_stop(hits_max_matrix, NDCG_max_matrix, MRR_max_matrix, patience_num_matrix,
                                            epoch_max_matrix,
                                            epoch, hits1, hits3, hits5, ndcg1, ndcg3, ndcg5, MRR_num)
            if patience_num_matrix[0][0] >= config.patience:
                break
    max_epoch = int(epoch_max_matrix[0][0])
    print('Saving train result：', result_list[max_epoch][1:])
    print('the optimal epoch', max_epoch)

    return result_list[max_epoch][1:]
