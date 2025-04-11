# -*- coding: utf-8 -*-
from dgl.ops import edge_softmax
from Utils.utils_ import *
import torch.nn.functional as F
from einops import repeat
from attention import aggregation
from Config import Config


class MessageAggregator(nn.Module):
    def __init__(self, num_heads, hidden_size, attn_drop, alpha, name, device):
        super(MessageAggregator, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = edge_softmax
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
        self.attn1 = nn.Linear(self.hidden_size, self.num_heads, bias=False).to(device)
        nn.init.xavier_normal_(self.attn1.weight, gain=1.414)
        self.attn2 = nn.Parameter(torch.empty(size=(1, self.num_heads, self.hidden_size), device=device))
        nn.init.xavier_normal_(self.attn2.data, gain=1.414)
        self.name = name
        self.device = device

    def forward(self, nodes, metapath_instances, metapath_embedding, features):
        device = self.device
        h_ = []
        for i in range(len(nodes)):
            index = metapath_instances[metapath_instances[self.name] == nodes[i]].index.tolist()
            if index != []:
                node_metapath_embedding = metapath_embedding[index].to(device)
                node_metapath_embedding = torch.cat([node_metapath_embedding] * self.num_heads, dim=1)
                node_metapath_embedding = node_metapath_embedding.unsqueeze(dim=0)
                eft = node_metapath_embedding.permute(1, 0, 2).view(-1, self.num_heads, self.hidden_size)
                node_embedding = torch.vstack([features[i]] * len(index)).to(device)
                a1 = self.attn1(node_embedding)
                a2 = (eft * self.attn2).sum(dim=-1)
                a = (a1 + a2).unsqueeze(dim=-1)
                a = self.leaky_relu(a)
                attention = F.softmax(a, dim=0)
                attention = self.attn_drop(attention)
                h = F.elu((attention * eft).sum(dim=0)).view(-1, self.hidden_size * self.num_heads)
                h_.append(h[0])
            else:
                node_embedding = torch.zeros(self.hidden_size * self.num_heads, device=device)
                h_.append(node_embedding)
        return torch.stack(h_, dim=0)


class Subgraph_Fusion(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Subgraph_Fusion, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1.414)

    def forward(self, z):
        device = z.device
        self.project.to(device)
        w = self.project(z).mean(0)
        beta_ = torch.softmax(w, dim=0)
        beta = beta_.expand((z.shape[0],) + beta_.shape)
        return (beta * z).sum(1), beta_


class SemanticEncoder(nn.Module):
    def __init__(self, layer_num_heads, hidden_size, r_vec, etypes, batch_size=8):
        super(SemanticEncoder, self).__init__()
        self.num_heads = layer_num_heads
        self.hidden_size = hidden_size
        self.r_vec = r_vec
        self.etypes = etypes
        self.batch_size = batch_size

    def forward(self, edata):
        edata = edata.reshape(edata.shape[0], edata.shape[1], edata.shape[2] // 2, 2)
        final_r_vec = torch.zeros([edata.shape[1], self.hidden_size // 2, 2], device=edata.device)
        r_vec = F.normalize(self.r_vec, p=2, dim=2)
        r_vec = torch.stack((r_vec, r_vec), dim=1)
        r_vec[:, 1, :, 1] = -r_vec[:, 1, :, 1]
        r_vec = r_vec.reshape(self.r_vec.shape[0] * 2, self.r_vec.shape[1], 2)
        final_r_vec[-1, :, 0] = 1
        for i in range(final_r_vec.shape[0] - 2, -1, -1):
            if self.etypes[i] is not None:
                final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 0] - \
                                       final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 1]
                final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 0].clone() * r_vec[self.etypes[i], :, 1] + \
                                       final_r_vec[i + 1, :, 1].clone() * r_vec[self.etypes[i], :, 0]
            else:
                final_r_vec[i, :, 0] = final_r_vec[i + 1, :, 0].clone()
                final_r_vec[i, :, 1] = final_r_vec[i + 1, :, 1].clone()
        for i in range(edata.shape[1] - 1):
            temp1 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 0] - \
                    edata[:, i, :, 1].clone() * final_r_vec[i, :, 1]
            temp2 = edata[:, i, :, 0].clone() * final_r_vec[i, :, 1] + \
                    edata[:, i, :, 1].clone() * final_r_vec[i, :, 0]
            edata[:, i, :, 0] = temp1
            edata[:, i, :, 1] = temp2
        edata = edata.reshape(edata.shape[0], edata.shape[1], -1)
        metapath_embedding = torch.mean(edata, dim=1)
        return metapath_embedding


class MTGNN_Layer(nn.Module):
    def __init__(self, meta_paths, test_data, hidden_size, r_vec, layer_num_heads, dropout, etypes, name, device):
        super(MTGNN_Layer, self).__init__()
        self.num_heads = layer_num_heads
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)
        self._cached_graph = None
        self._cached_coalesced_graph = {}
        self.r_vec = r_vec
        self.etypes = etypes
        self.message_aggregator_layer = nn.ModuleList()
        self.semantic_encoder_layer = nn.ModuleList()
        self.hidden_size = hidden_size
        self.test_data = test_data
        # # embedding
        for i in range(len(meta_paths)):
            self.semantic_encoder_layer.append(
                SemanticEncoder(self.num_heads, self.hidden_size, self.r_vec, self.etypes[i]))
        for i in name:
            self.message_aggregator_layer.append(
                MessageAggregator(self.num_heads, self.hidden_size, attn_drop=dropout, alpha=0.01, name=i, device=device))
        self.subgraph_fusion = Subgraph_Fusion(in_size=self.hidden_size * self.num_heads)
        self.separate_metapath_subgraph = Separate_subgraph(device=device)
        self.exclude_test = Prevent_leakage(self.test_data)

    def stack_embedding(self, embeddings):
        subgraph_num_nodes = [embeddings[i].size()[0] for i in range(len(embeddings))]
        if subgraph_num_nodes.count(subgraph_num_nodes[0]) == len(subgraph_num_nodes):
            embeddings = torch.stack(embeddings, dim=1)
        else:
            for i in range(0, len(embeddings)):
                index = max(subgraph_num_nodes) - subgraph_num_nodes[i]
                if index != 0:
                    h_ = torch.zeros(index, self.hidden_size * self.num_heads)
                    embeddings[i] = torch.cat((embeddings[i], h_), dim=0)
            embeddings = torch.stack(embeddings, dim=1)
        return embeddings

    def generate_metapath_instances(self, g, meta_path, batch_size=10000):
        edges = [g.edges(etype=f"{meta_path[j]}_{meta_path[j + 1]}") for j in range(len(meta_path) - 1)]
        edges = [[edges[i][j].tolist() for j in range(len(edges[i]))] for i in range(len(edges))]

        df_0 = pd.DataFrame(edges[0], index=list(meta_path)[:2]).T
        df_1 = pd.DataFrame(edges[1], index=list(meta_path)[-2:]).T
        num_batches = (len(df_0) + batch_size - 1) // batch_size
        metapath_instances_list = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df_0))
            df_0_batch = df_0.iloc[start_idx:end_idx]
            metapath_instances_batch = pd.merge(df_0_batch, df_1, how='inner')
            filt_metapath_instances_batch = metapath_instances_batch[['g', 't', 'd']]
            filt_metapath_instances_batch = self.exclude_test(filt_metapath_instances_batch)
            metapath_instances_list.append(filt_metapath_instances_batch[list(meta_path)])
        metapath_instances = pd.concat(metapath_instances_list, ignore_index=True)
        return metapath_instances


    def forward(self, g, h):
        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[meta_path] = self.separate_metapath_subgraph(g, meta_path)

        semantic_embeddings = {'g': [], 't': [], 'd': []}
        nodes_embeddings = {}
        batch_size = 1

        for i, meta_path in (enumerate(self.meta_paths)):
            edata_list = []
            config = Config()
            new_g = self._cached_coalesced_graph[meta_path]
            metapath_instances = self.generate_metapath_instances(new_g, meta_path)

            max_index = -1
            for j in range(len(meta_path)):
                input_tensor = torch.tensor(metapath_instances.iloc[:, j].values, dtype=torch.long).to(config.device)
                max_index = max(max_index, input_tensor.max().item())
                if meta_path[j] == 'g':
                    embedding_weight = h['g']
                elif meta_path[j] == 't':
                    embedding_weight = h['t']
                elif meta_path[j] == 'd':
                    embedding_weight = h['d']
                else:
                    raise ValueError(f"Invalid node type: {meta_path[j]}")

                input_tensor = torch.clamp(input_tensor, max=embedding_weight.size(0) - 1)

                if input_tensor.max().item() >= embedding_weight.size(0):
                    raise ValueError(f"Index out of range: {input_tensor.max().item()} >= {embedding_weight.size(0)}")

                edata = F.embedding(input_tensor, embedding_weight).unsqueeze(1)
                edata_list.append(edata)

            for k in range(0, len(edata_list), batch_size):
                batch_data = edata_list[k:k + batch_size]
                batch_metapathembedding = []

                for edata in batch_data:
                    metapathembedding = self.semantic_encoder_layer[i](edata)
                    batch_metapathembedding.append(metapathembedding)

            semantic_embeddings['g'].append(
                self.message_aggregator_layer[0](new_g.nodes('g').tolist(), metapath_instances, metapathembedding,
                                                 h['g']))
            semantic_embeddings['t'].append(
                self.message_aggregator_layer[1](new_g.nodes('t').tolist(), metapath_instances, metapathembedding,
                                                 h['t']))
            semantic_embeddings['d'].append(
                self.message_aggregator_layer[2](new_g.nodes('d').tolist(), metapath_instances, metapathembedding,
                                                 h['d']))

        for ntype in semantic_embeddings.keys():
            if ntype == 'g':
                semantic_embeddings[ntype] = self.stack_embedding(semantic_embeddings[ntype])
                nodes_embeddings[ntype], g_beta = self.subgraph_fusion(semantic_embeddings[ntype])
            elif ntype == 't' and semantic_embeddings[ntype]:
                semantic_embeddings[ntype] = self.stack_embedding(semantic_embeddings[ntype])
                nodes_embeddings[ntype], m_beta = self.subgraph_fusion(semantic_embeddings[ntype])
            elif ntype == 'd' and semantic_embeddings[ntype]:
                semantic_embeddings[ntype] = self.stack_embedding(semantic_embeddings[ntype])
                nodes_embeddings[ntype], d_beta = self.subgraph_fusion(semantic_embeddings[ntype])

        return nodes_embeddings


class Common_model(nn.Module):
    def __init__(self, config):
        super(Common_model, self).__init__()

        self.device = config.device

        self.conv = nn.Conv2d(config.num_filters, config.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom

        self.smi_emb = nn.Embedding(config.smi_dict_len + 1, config.embedding_size)
        self.smi_conv_region = nn.Conv2d(1, config.num_filters, (3, config.embedding_size), stride=1, padding=(1, 0))

        self.fas_L1 = nn.Linear(config.fasta_max_len, config.common_size)
        self.fas_bn1 = nn.BatchNorm1d(config.common_size)

        self.mesh_emb = nn.Embedding(config.mesh_dict_len + 1, config.embedding_size)
        self.mesh_conv_region = nn.Conv2d(1, config.num_filters, (3, config.embedding_size), stride=1, padding=(1, 0))

        self.smi_mlp = MLP(config.num_filters, config.common_size)
        # self.fas_mlp = MLP(config.num_filters, config.common_size)
        self.mesh_mlp = MLP(config.num_filters, config.common_size)

        self.to(self.device)

    def forward(self, smiles, fasta, mesh):

        # print("Max index in smiles:", smiles.max().item())
        # print("Min index in smiles:", smiles.min().item())

        smiles_vector = self.smi_emb(smiles)

        smiles_vector = torch.unsqueeze(smiles_vector, 1)

        smiles_vector = self.smi_conv_region(smiles_vector)
        # Repeat 2 times
        smiles_vector = self.padding1(smiles_vector)
        smiles_vector = torch.relu(smiles_vector)
        smiles_vector = self.conv(smiles_vector)
        smiles_vector = self.padding1(smiles_vector)
        smiles_vector = torch.relu(smiles_vector)
        smiles_vector = self.conv(smiles_vector)

        while smiles_vector.size()[2] >= 2:
            smiles_vector = self._block(smiles_vector)
        smiles_vector = smiles_vector.squeeze()
        smile_common = self.smi_mlp(smiles_vector)

        # print("Max index in fasta:", fasta.max().item())
        # print("Min index in fasta:", fasta.min().item())

        fasta_vector = self.fas_L1(fasta)
        fasta_common = self.fas_bn1(F.relu(fasta_vector))

        mesh_vector = self.mesh_emb(mesh)
        mesh_vector = torch.unsqueeze(mesh_vector, 1)
        mesh_vector = self.mesh_conv_region(mesh_vector)
        # Repeat 2 times
        mesh_vector = self.padding1(mesh_vector)
        mesh_vector = torch.relu(mesh_vector)
        mesh_vector = self.conv(mesh_vector)
        mesh_vector = self.padding1(mesh_vector)
        mesh_vector = torch.relu(mesh_vector)
        mesh_vector = self.conv(mesh_vector)

        while mesh_vector.size()[2] >= 2:
            mesh_vector = self._block(mesh_vector)
        mesh_vector = mesh_vector.squeeze()
        mesh_common = self.mesh_mlp(mesh_vector)

        return smile_common, fasta_common, mesh_common

    def _block(self, x):

        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x

# multi-layer perceptron
class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(input_size // 2, output_size)
        )

    def forward(self, x):
        out = self.linear(x)
        return out

class MTGNN(nn.Module):
    def __init__(self, meta_paths, test_data, in_size, hidden_size, num_heads, dropout, etypes, config, device):
        super(MTGNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.device = device
        self.fc_g = nn.Linear(in_size['g'], hidden_size).to(self.device)
        self.fc_m = nn.Linear(in_size['t'], hidden_size).to(self.device)
        self.fc_d = nn.Linear(in_size['d'], hidden_size).to(self.device)
        self.fc_common = nn.Linear(hidden_size * num_heads * 3, config.common_size).to(self.device)
        self.smile_weight = nn.Parameter(torch.Tensor([0.5])).to(self.device)
        self.fasta_weight = nn.Parameter(torch.Tensor([0.5])).to(self.device)
        self.mesh_weight = nn.Parameter(torch.Tensor([0.5])).to(self.device)
        self.predict = nn.Sequential(
            nn.Linear(self.hidden_size * self.num_heads * 3, self.hidden_size * self.num_heads),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * self.num_heads, self.hidden_size * self.num_heads // 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * self.num_heads // 4, self.hidden_size),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 1)
        ).to(self.device)
        r_vec = nn.Parameter(torch.empty(size=(3, self.hidden_size // 2, 2)))
        self.layers1 = MTGNN_Layer(meta_paths, test_data, hidden_size, r_vec, num_heads, dropout, etypes,
                                    name=['g', 't', 'd'], device=self.device)
        self.predict.apply(self.weights_init)
        self.common = Common_model(config)
        self.aggregation = aggregation(depth=1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size)).to(self.device)
        # Multihead attention for cross-modal aggregation
        self.attn = nn.MultiheadAttention(embed_dim=config.common_size, num_heads=config.num_heads,
                                          dropout=config.dropout, device=self.device)
        nn.init.xavier_normal_(self.fc_g.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_m.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc_d.weight, gain=1.414)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1.414)

    def get_embed_map(self, features, embed_features, data):
        embed_feature_keys = list(embed_features.keys())
        stack_embedding = {'g': [], 't': [], 'd': []}
        stack_embedding_keys = ['g', 't', 'd']

        for i, embed_key in enumerate(embed_feature_keys):
            embed_length = len(embed_features[embed_key])

            for j in range(len(data)):
                value = data[j, i].item()

                if 0 <= value < embed_length:
                    embedding_vector = embed_features[embed_key][value]
                else:
                    id = int(value)
                    embedding_vector = torch.hstack([features[embed_key][id]] * 8)

                    target_dim = 256
                    if embedding_vector.shape[0] < target_dim:
                        padding = target_dim - embedding_vector.shape[0]
                        embedding_vector = F.pad(embedding_vector, (0, padding))
                    elif embedding_vector.shape[0] > target_dim:
                        embedding_vector = embedding_vector[:target_dim]

                stack_embedding[stack_embedding_keys[i]].append(embedding_vector)

            stack_embedding[stack_embedding_keys[i]] = torch.stack(
                stack_embedding[stack_embedding_keys[i]], dim=0)

        embedding_concat = torch.cat((stack_embedding['g'], stack_embedding['t'], stack_embedding['d']), dim=1)

        return embedding_concat

    def forward(self, g, inputs, data, smiles, fasta, mesh):
        h_trans = {}
        h_trans['g'] = self.fc_g(inputs['g']).view(-1, self.hidden_size)
        h_trans['t'] = self.fc_m(inputs['t']).view(-1, self.hidden_size)
        h_trans['d'] = self.fc_d(inputs['d']).view(-1, self.hidden_size)

        smile_common, fasta_common, mesh_common = self.common(smiles, fasta, mesh)

        fused_feature = {}
        v_d = h_trans['g'].unsqueeze(1)
        v_p = smile_common.unsqueeze(1)
        bs = v_p.size(0)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=bs)
        fused_feature['g'] = self.aggregation(v_p, v_d, cls_tokens)

        v_d = h_trans['t'].unsqueeze(1)
        v_p = fasta_common.unsqueeze(1)
        bs = v_p.size(0)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=bs)
        fused_feature['t'] = self.aggregation(v_p, v_d, cls_tokens)

        v_d = h_trans['d'].unsqueeze(1)
        v_p = mesh_common.unsqueeze(1)
        bs = v_p.size(0)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=bs)
        fused_feature['d'] = self.aggregation(v_p, v_d, cls_tokens)

        h_trans_embed = self.layers1(g, fused_feature)
        h_concat = self.get_embed_map(fused_feature, h_trans_embed, data)
        predict_score = torch.sigmoid(self.predict(h_concat))

        return predict_score


