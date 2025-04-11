# -*- coding: utf-8 -*-
import warnings
from Utils.utils_ import *
from sklearn.model_selection import KFold
from Config import Config
warnings.filterwarnings("ignore")

def transfer_features_to_cuda(features):
    config = Config()
    for key in features:
        features[key] = features[key].to(config.device)
    return features


def load_sequences(file_path, max_len):
    with open(file_path, 'r') as f:
        data = f.readlines()

    sequences = [seq.strip()[:max_len] for seq in data]

    padded_sequences = []
    for seq in sequences:
        ascii_seq = [ord(char) for char in seq]
        if len(ascii_seq) < max_len:
            ascii_seq += [0] * (max_len - len(ascii_seq))
        padded_sequences.append(ascii_seq)

    tensor_data = torch.tensor(padded_sequences, dtype=torch.long)
    return tensor_data

def generate_protein( ):
    config = Config()
    LLM_path = config.pt_fasta_path
    embedding_dim = config.fasta_max_len
    embeddings = []

    with open(LLM_path, 'r') as f:
        for line in f:

            protein_embedding = list(map(float, line.strip().split()))

            current_dim = len(protein_embedding)

            if current_dim > embedding_dim:
                protein_embedding = protein_embedding[:embedding_dim]
            elif current_dim < embedding_dim:
                padding = [0] * (embedding_dim - current_dim)
                protein_embedding.extend(padding)

            embeddings.append(torch.tensor(protein_embedding))

    protein_tensor = torch.stack(embeddings)
    protein_tensor = protein_tensor.to(config.device)

    return protein_tensor

def load_sequences_features(config):
    smiles = load_sequences(config.dg_smiles_path, config.smiles_max_len).to(config.device)
    fasta = generate_protein( )
    mesh = load_sequences(config.di_mesh_path, config.mesh_max_len).to(config.device)

    print("SMILES data shape:", smiles.shape)
    print("FASTA data shape:", fasta.shape)
    print("mesh data shape:", mesh.shape)

    return smiles, fasta, mesh


def data_lode():
    'Read the initial features files'
    protein_feat = pd.read_csv('./Data/kinase_protein_matrix_3.txt', header=None, sep='\s+').values
    drug_feat = pd.read_csv('./Data/kinase_drug_matrix_3.txt', header=None, sep='\s+').values
    disease_feat = pd.read_csv('./Data/kinase_disease_matrix_3.txt', header=None, sep='\s+').values

    drug_features = torch.FloatTensor(drug_feat)
    disease_features = torch.FloatTensor(disease_feat)
    protein_features = torch.FloatTensor(protein_feat)

    features = {'g': drug_features, 't': protein_features, 'd': disease_features}
    in_size = {'g': drug_features.shape[1], 't': protein_features.shape[1],
               'd': disease_features.shape[1]}
    print("Features Loaded:")
    print("Drug features shape:", drug_features.shape)
    print("protein features shape:", protein_features.shape)
    print("Disease features shape:", disease_features.shape)
    features = transfer_features_to_cuda(features)
    return features, in_size



def get_train_val_data(all_data, train_ind, val_ind, adj, seed):
    'To generate negative samples for the training set for 5-fold CV'
    neg_num_test = 10
    train_data_pos, val_data_pos = all_data[train_ind], all_data[val_ind]
    print("Train Positive Data Shape:", train_data_pos.shape)
    print("Validation Positive Data Shape:", val_data_pos.shape)
    train_data_all, tr_neg_1_ls, tr_neg_2_ls, tr_neg_3_ls, te_neg_1_ls, te_neg_2_ls, te_neg_3_ls = neg_data_generate(
        adj, train_data_pos, val_data_pos, neg_num_test, seed)
    np.random.shuffle(train_data_all)
    val_neg_data = []
    for i in te_neg_1_ls:
        if list(i)[-1] == 0:
            val_neg_data.append(i)
    train_data_pos = train_data_pos.copy().astype(int)
    val_data_pos = val_data_pos.copy().astype(int)
    return train_data_pos, np.array(tr_neg_1_ls), val_data_pos, np.array(val_neg_data)


def get_indep_data(adj, train_data_pos, val_data_pos, seed):
    'To generate negative samples for the training set for independent test'
    neg_num_test = 10
    train_data_all, tr_neg_1_ls, tr_neg_2_ls, tr_neg_3_ls, te_neg_1_ls, te_neg_2_ls, te_neg_3_ls = neg_data_generate(
        adj, train_data_pos, val_data_pos, neg_num_test, seed)
    np.random.shuffle(train_data_all)
    val_neg_data = []
    for i in te_neg_1_ls:
        if list(i)[-1] == 0:
            val_neg_data.append(i)
    train_data_pos = train_data_pos.copy().astype(int)
    val_data_pos = val_data_pos.copy().astype(int)
    return train_data_pos, tr_neg_1_ls, val_data_pos, np.array(val_neg_data)


def neg_data_generate(adj_data_all, train_data_fix, val_data_fix, neg_num_test, seed):
    'A function used to generate negative samples'
    neg_num_train = 1
    np.random.seed(seed)
    train_neg_1_ls = []
    train_neg_2_ls = []
    train_neg_3_ls = []
    val_neg_1_ls = []
    val_neg_2_ls = []
    val_neg_3_ls = []
    max_x = int(np.max(adj_data_all[:, 0])) + 1
    max_y = int(np.max(adj_data_all[:, 1])) + 1
    max_z = int(np.max(adj_data_all[:, 2])) + 1
    arr_true = np.zeros((max_x, max_y, max_z),dtype=np.int16)
    for line in adj_data_all:
        arr_true[int(line[0]), int(line[1]), int(line[2])] = 1
    arr_false_train = np.zeros((max_x, max_y, max_z), dtype=np.int16)

    print("True Array Shape:", arr_true.shape)
    print("Max indices:", max_x, max_y, max_z)

    for i in train_data_fix:
        ctn_1 = 0
        ctn_2 = 0
        ctn_3 = 0
        tr_drug_ls = [j for j in range(0, arr_true.shape[0])]
        tr_pro_ls = [j for j in range(0, arr_true.shape[1])]
        tr_dis_ls = [j for j in range(0, arr_true.shape[2])]

        while ctn_1 < neg_num_train:
            a = np.random.randint(0, arr_true.shape[0] - 1)  # random select a drug
            b = np.random.randint(0, arr_true.shape[1] - 1)
            c = np.random.randint(0, arr_true.shape[2] - 1)
            if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                arr_false_train[a, b, c] = 1
                ctn_1 += 1
                train_neg_1_ls.append((a, b, c, 0))
        while ctn_2 < neg_num_train:
            b = int(i[1])
            c = int(i[2])
            a = np.random.choice(tr_drug_ls)
            if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                arr_false_train[a, b, c] = 1
                ctn_2 += 1
                train_neg_2_ls.append((a, b, c, 0))
        while ctn_3 < neg_num_train:
            a = int(i[0])
            b = np.random.randint(0, arr_true.shape[1] - 1)
            c = int(i[2])
            if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
                arr_false_train[a, b, c] = 1
                ctn_3 += 1
                train_neg_3_ls.append((a, b, c, 0))

    train_neg_1_arr = np.array(train_neg_1_ls)
    train_neg_2_arr = np.array(train_neg_2_ls)
    train_neg_3_arr = np.array(train_neg_3_ls)
    # train_neg_4_arr = np.array(train_neg_4_ls)
    train_neg_all = np.vstack(
        (np.vstack((np.vstack((np.vstack((train_neg_1_arr, train_neg_2_arr)), train_neg_3_arr)))),
         train_data_fix))
    np.random.shuffle(train_neg_all)

    print("Final Train Neg All Shape:", train_neg_all.shape)

    for i in val_data_fix:
        neg_1_i = []
        neg_2_i = []
        neg_3_i = []
        neg_4_i = []
        # Because it is too easy to repeat, it is only guaranteed that multiple negative samples generated for a
        # certain positive sample are not repeated, and negative samples of different positive samples may be repeated.
        arr_false_val_1 = np.zeros(arr_true.shape, dtype=np.int16)
        arr_false_val_2 = np.zeros(arr_true.shape, dtype=np.int16)
        arr_false_val_3 = np.zeros(arr_true.shape, dtype=np.int16)
        # arr_false_val_4 = np.zeros(arr_true.shape, dtype=np.int16)
        neg_1_i.append(i)
        neg_2_i.append(i)
        neg_3_i.append(i)
        neg_4_i.append(i)
        cva_1 = 0
        cva_2 = 0
        cva_3 = 0
        # cva_4 = 0
        drug_ls = [j for j in range(0, arr_true.shape[0])]
        pro_ls = [j for j in range(0, arr_true.shape[1])]
        dis_ls = [j for j in range(0, arr_true.shape[2])]
        while cva_1 < neg_num_test:
            a_1 = np.random.randint(0, arr_true.shape[0] - 1)
            b_1 = np.random.randint(0, arr_true.shape[1] - 1)
            c_1 = np.random.randint(0, arr_true.shape[2] - 1)
            if arr_true[a_1, b_1, c_1] != 1 and arr_false_train[a_1, b_1, c_1] != 1 and arr_false_val_1[
                a_1, b_1, c_1] != 1:
                arr_false_val_1[a_1, b_1, c_1] = 1
                cva_1 += 1
                neg_1_i.append((a_1, b_1, c_1, 0))
        np.random.shuffle(neg_1_i)
        val_neg_1_ls.extend(neg_1_i)
        while cva_2 < neg_num_test:
            b_2 = int(i[1])
            c_2 = int(i[2])
            if drug_ls != []:
                a_2 = np.random.choice(drug_ls)
                drug_ls.remove(a_2)
                if arr_true[a_2, b_2, c_2] != 1 and arr_false_train[a_2, b_2, c_2] != 1 and arr_false_val_2[
                    a_2, b_2, c_2] != 1:
                    arr_false_val_2[a_2, b_2, c_2] = 1
                    cva_2 += 1
                    neg_2_i.append((a_2, b_2, c_2, 0))
            else:
                distance_2 = neg_num_test - cva_2
                last_ind = len(neg_2_i) - 1
                for k in range(distance_2):
                    neg_2_i.append(neg_2_i[last_ind])
                break
        np.random.shuffle(neg_2_i)
        val_neg_2_ls.extend(neg_2_i)

        while cva_3 < neg_num_test:
            a_3 = int(i[0])
            c_3 = int(i[2])
            if pro_ls != []:
                b_3 = np.random.choice(pro_ls)
                pro_ls.remove(b_3)
                if arr_true[a_3, b_3, c_3] != 1 and arr_false_train[a_3, b_3, c_3] != 1 and arr_false_val_3[
                    a_3, b_3, c_3] != 1:
                    arr_false_val_3[a_3, b_3, c_3] = 1
                    cva_3 += 1
                    neg_3_i.append((a_3, b_3, c_3, 0))
            else:
                distance_3 = neg_num_test - cva_3
                last_ind = len(neg_3_i) - 1
                for k in range(distance_3):
                    neg_3_i.append(neg_3_i[last_ind])
                break
        np.random.shuffle(neg_3_i)
        val_neg_3_ls.extend(neg_3_i)

    return train_neg_all, train_neg_1_ls, train_neg_2_ls, train_neg_3_ls, val_neg_1_ls, val_neg_2_ls, val_neg_3_ls


def generate_dataset():
    config = Config()
    np.random.seed(config.seed)
    adj_data = np.loadtxt('./Data/d_p_d_pairs_kinase_sample_2.txt')
    np.random.shuffle(adj_data)
    cv_data = adj_data[int(0.1 * len(adj_data)):, :]
    indep_data = adj_data[:int(0.1 * len(adj_data)), :]
    fold_num = 0
    kf = KFold(n_splits=5, shuffle=True, random_state=config.seed)
    for train_index, val_index in kf.split(cv_data):
        fold_num += 1
        train_data_pos, train_data_neg, val_data_pos, val_data_neg = get_train_val_data(cv_data, train_index, val_index,
                                                                                        adj_data, config.seed)
        np.savetxt('./Data/CV_data/CV_' + str(fold_num) + '/train_data_pos.csv', train_data_pos, delimiter=",",
                   fmt='%d')
        np.savetxt('./Data/CV_data/CV_' + str(fold_num) + '/train_data_neg.csv', train_data_neg, delimiter=',',
                   fmt='%d')
        np.savetxt('./Data/CV_data/CV_' + str(fold_num) + '/val_data_pos.csv', val_data_pos, delimiter=',', fmt='%d')
        np.savetxt('./Data/CV_data/CV_' + str(fold_num) + '/val_data_neg.csv', val_data_neg, delimiter=',', fmt='%d')
    train_data_pos, train_data_neg, test_data_pos, test_data_neg = get_indep_data(adj_data, cv_data, indep_data,
                                                                                  config.seed)
    np.savetxt('./Data/indepent_data/train_data_pos.csv', train_data_pos, delimiter=",", fmt='%d')
    np.savetxt('./Data/indepent_data/train_data_neg.csv', train_data_neg, delimiter=",", fmt='%d')
    np.savetxt('./Data/indepent_data/test_data_pos.csv', test_data_pos, delimiter=",", fmt='%d')
    np.savetxt('./Data/indepent_data/test_data_neg.csv', test_data_neg, delimiter=",", fmt='%d')

'''
If you want to regenerate the training set and validation set data, 
you can uncomment this line of code and regenerate the data'
'''
# generate_dataset()