# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import random
import dgl
import pandas as pd
from tqdm import tqdm


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Myloss(nn.Module):
    def __init__(self, device):
        super(Myloss, self).__init__()
        self.device = device
        self.eps = torch.tensor(1e-5).to(device)

    def forward(self, iput, target, gamma):
        loss_sum = torch.pow((iput - target), 2)
        result = (1 - gamma) * ((target * loss_sum).sum()) + gamma * (((1 - target) * loss_sum).sum())
        return (result + self.eps)


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.414)

class Matrix(nn.Module):
    def __init__(self):
        super(Matrix, self).__init__()

    def hits(self, pos_index, scores_index):
        if len(pos_index) == 0:
            return 0
        pos_value = pos_index[0].item()

        scores_index = torch.tensor(scores_index, dtype=torch.long)

        if pos_value in scores_index:
            Hits = 1
        else:
            Hits = 0

        return Hits

    def ndcg(self, pos_index, scores_index, n):
        if len(pos_index) == 0:
            return 0, 0
        dcg_sum = 0
        idcg_sum = 0
        for j in range(len(scores_index)):
            if scores_index[j] == pos_index[0]:
                dcg_sum += self.dcg(1, j + 1)
            else:
                dcg_sum += self.dcg(0, j + 1)
        for m in range(n):
            idcg_sum += self.dcg(1 if m == 0 else 0, m + 1)
        return dcg_sum, idcg_sum

    def dcg(self, rel, index):
        return (2 ** rel - 1) / torch.log2(torch.tensor(index + 1, dtype=torch.float32))

    def forward(self, n, num, predict_val, num_pos, index):
        sample_hit, sample_ndcg = [], []
        Hits_sum = 0
        ndcg_sum = 0

        index_tuple = sorted(enumerate(index), reverse=False, key=lambda index: index[1])
        index_list = [index[0] for index in index_tuple]
        predict_val = predict_val[index_list]

        for i in range(num_pos):
            neg_scores = predict_val[num_pos + i * num:num_pos + (i + 1) * num]
            scores = torch.cat([neg_scores, predict_val[i].unsqueeze(0)])
            random_num = torch.randperm(scores.size(0))
            if num >= scores.size(0):
                print(f"Warning: num ({num}) exceeds scores size ({scores.size(0)})")
                pos_index = torch.tensor([])
            else:
                random_num[-1] = num
                pos_index = (random_num == num).nonzero(as_tuple=True)[0]
            scores = scores[random_num]
            scores_tuple = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            scores_index = [scores[0] for scores in scores_tuple[:n]]

            if len(pos_index) == 0:
                print(f"Warning: pos_index is empty for sample {i}, skipping this sample.")
                continue

            Hits = self.hits(pos_index, scores_index)
            dcg_sum, idcg_sum = self.ndcg(pos_index, scores_index, n)
            if idcg_sum > 0:
                ndcg_sum += dcg_sum / idcg_sum
            Hits_sum += Hits
            sample_hit.append(Hits)
            sample_ndcg.append(dcg_sum / idcg_sum if idcg_sum > 0 else 0)

        Hits = Hits_sum / num_pos
        ndcg = ndcg_sum / num_pos
        return Hits, ndcg, sample_hit, sample_ndcg


class MRR(nn.Module):
    def __init__(self):
        super(MRR, self).__init__()


    def forward(self, num, predict_val, num_pos, index):
        sample_mrr = []
        rank_sum = 0

        index = index.to(predict_val.device)

        index_tuple = sorted(enumerate(index), reverse=False, key=lambda index: index[1])
        index_list = [index[0] for index in index_tuple]
        predict_val = predict_val[index_list]

        for i in range(num_pos):
            neg_scores = predict_val[num_pos + i * num:num_pos + (i + 1) * num]
            scores = torch.cat([neg_scores, predict_val[i].unsqueeze(0)])

            random_num = torch.randperm(scores.size(0), device=predict_val.device)  # 确保在同一设备上
            pos_index = (random_num == num).nonzero(as_tuple=True)[0]
            scores = scores[random_num]

            scores_tuple = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            scores_index = [scores[0] for scores in scores_tuple]

            if len(scores_index) == 0:
                sample_mrr.append(0)
                rank_sum += 0
                continue

            try:
                rank = scores_index.index(pos_index[0].item()) + 1
                sample_mrr.append(1 / rank)
                rank_sum += 1 / rank
            except ValueError:
                sample_mrr.append(0)
                rank_sum += 0

        mrr = rank_sum / num_pos
        return mrr, sample_mrr


def construct_hg(pos_data,device):
    g_m_edges, m_d_edges, g_d_edges = [], [], []
    g_m_set, m_d_set, g_d_set = set(), set(), set()

    for i in range(len(pos_data)):
        one_g_m_edge = tuple(pos_data[i][0:2].tolist())
        one_m_d_edge = tuple(pos_data[i][1:3].tolist())
        one_g_d_edge = (pos_data[i][0], pos_data[i][2])

        if one_g_m_edge not in g_m_set:
            g_m_edges.append(one_g_m_edge)
            g_m_set.add(one_g_m_edge)
        if one_m_d_edge not in m_d_set:
            m_d_edges.append(one_m_d_edge)
            m_d_set.add(one_m_d_edge)
        if one_g_d_edge not in g_d_set:
            g_d_edges.append(one_g_d_edge)
            g_d_set.add(one_g_d_edge)

    g_m_edges = torch.tensor(sorted(g_m_edges, key=lambda x: x[0]), dtype=torch.long).to(device)
    m_d_edges = torch.tensor(sorted(m_d_edges, key=lambda x: x[0]), dtype=torch.long).to(device)
    g_d_edges = torch.tensor(sorted(g_d_edges, key=lambda x: x[0]), dtype=torch.long).to(device)

    hg = dgl.heterograph({
        ('g', 'g_t', 't'): (g_m_edges[:, 0], g_m_edges[:, 1]),
        ('t', 't_d', 'd'): (m_d_edges[:, 0], m_d_edges[:, 1]),
        ('g', 'g_d', 'd'): (g_d_edges[:, 0], g_d_edges[:, 1]),
        ('t', 't_g', 'g'): (g_m_edges[:, 1], g_m_edges[:, 0]),
        ('d', 'd_t', 't'): (m_d_edges[:, 1], m_d_edges[:, 0]),
        ('d', 'd_g', 'g'): (g_d_edges[:, 1], g_d_edges[:, 0])
    })

    return hg


class Prevent_leakage(nn.Module):
    def __init__(self, test_data):
        super(Prevent_leakage, self).__init__()
        self.test_data = test_data

    def forward(self, metapath_instances):
        test_pos_data = pd.DataFrame(self.test_data[:, :3].cpu().numpy(), columns=['g', 't', 'd'])
        metapath_instances_all = pd.concat([metapath_instances, test_pos_data], ignore_index=True)
        exclude_metapath_instances = metapath_instances_all.drop_duplicates(subset=['g', 't', 'd'], keep=False)
        exclude_metapath_instances = exclude_metapath_instances.reset_index(drop=True)
        return exclude_metapath_instances


class Separate_subgraph(nn.Module):
    def __init__(self, device):
        super(Separate_subgraph, self).__init__()
        self.device = device

    def get_edges(self, edges1, edges2):
        new_edges = [[list() for j in range(2)] for i in range(2)]
        for i in tqdm(range(len(edges1[0])),ncols=100):
            if edges1[1][i] in edges2[0]:
                new_edges[0][0].append(edges1[0][i])
                new_edges[0][1].append(edges1[1][i])
                index = [m for m, x in enumerate(edges2[0]) if x == edges1[1][i]]
                if edges1[1][i] not in new_edges[1][0]:
                    for j in range(len(index)):
                        new_edges[1][0].append(edges1[1][i])
                        new_edges[1][1].append(edges2[1][index[j]])
        return new_edges

    def forward(self, hg, metapath):
        new_triplets_edge = []
        metapath_list = [f"{metapath[i]}_{metapath[i + 1]}" for i in range(len(metapath) - 1)]
        edges = [hg.edges(etype=metapath_list[i]) for i in range(len(metapath_list))]
        edges = [[edges[i][j].tolist() for j in range(len(edges[i]))] for i in
                 range(len(edges))]
        if len(metapath_list) == 2:

            new_edges = self.get_edges(edges[0], edges[1])

        elif len(metapath_list) == 3:

            new_edges = self.get_edges(edges[0], edges[1])
            new_edges1 = self.get_edges(new_edges[1], edges[2])
            new_edges.append(new_edges1[1])

        for path in metapath_list:
            for i in range(len(hg.canonical_etypes)):
                if path in hg.canonical_etypes[i]:
                    new_triplets_edge.append(hg.canonical_etypes[i])
        graph_data = {}
        for i in range(len(metapath_list)):
            graph_data[new_triplets_edge[i]] = (new_edges[i][0], new_edges[i][1])
        subgraph = dgl.heterograph(graph_data)
        subgraph = subgraph.to(self.device)

        return subgraph


def ealy_stop(hits_max_matrix, NDCG_max_matrix, MRR_max_matrix, patience_num_matrix, epoch_max_matrix, e, hits_1,
              hits_3, hits_5, ndcg1, ndcg3, ndcg5, MRR):
    if hits_1 >= hits_max_matrix[0][0]:
        hits_max_matrix[0][0] = hits_1
        hits_max_matrix[0][1] = hits_3
        hits_max_matrix[0][2] = hits_5
        NDCG_max_matrix[0][0] = ndcg1
        NDCG_max_matrix[0][1] = ndcg3
        NDCG_max_matrix[0][2] = ndcg5
        MRR_max_matrix[0][0] = MRR
        epoch_max_matrix[0][0] = e
        patience_num_matrix[0][0] = 0
    else:
        patience_num_matrix[0][0] += 1
    return patience_num_matrix


