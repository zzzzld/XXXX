import torch

class Config:
    def __init__(self):
        # General model parameters
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Device to use
        # self.use_gpu = True
        self.lr = 0.0075                        # Learning rate
        self.seed = 111                         # Random seed
        self.num_epochs = 10                    # Number of training epochs100
        self.dropout = 0.5                      # Dropout rate
        self.weight_decay = 0.01                # Weight decay
        self.loss_gamma = 0.6                  # Balance coefficient for loss
        self.hidden_size = 32                   # Model hidden dimension8
        self.num_heads = 8                     # Number of attention heads in GAT
        self.patience = 20                     # Patience for early stopping
        self.repeat_nums = 1                   # Repeat numbers for training
        self.fold_nums = 10                    # Cross-validation folds
        self.neg_samp_ratio = 1               # Negative to positive sample ratio

        # Metapath and edge types
        self.etypes = [[0, 1], [2, 3], [4, 0], [5, 2], [3, 5], [1, 4]]  # Edge types in metapaths
        self.metapaths = [['g', 't', 'd'], ['t', 'g', 'd'], ['d', 'g', 't'], ['d', 't', 'g'], ['g', 'd', 't'], ['t', 'd', 'g']]  # Types of metapaths

        # Dataset-related parameters
        self.smi_n_gram = 1
        self.fas_n_gram = 3
        self.smi_dict_len = 118               # Length of SMILES dictionary
        self.fas_dict_len = 90                # Length of FASTA dictionary
        self.mesh_dict_len = 69               # Length of MESH dictionary
        self.fasta_max_len = 2500             # Max length of protein sequences
        self.smiles_max_len = 1500            # Max length of SMILES sequences
        self.mesh_max_len = 1000             # Max length of MESH sequences
        # self.ds_nums = 5603
        # self.se_nums = 4192

        # Path to datasets
        self.dg_smiles_path = './Data/smiles.txt'
        self.pt_fasta_path = './Data/protein_embeddings_batch_256.txt'
        self.di_mesh_path = './Data/disease.txt'

        # Neural network-related parameters
        self.embedding_size = 64              # Embedding size
        self.num_filters = 128                # Number of filters for convolutional layers
        self.common_size = 32                  # Common layer size
        self.common_learn_rate = 0.00001      # Common learning rate
        self.pre_learn_rate = 0.00001         # Pre-training learning rate

        # Other parameters
        self.common_epochs = 1                # Number of common epochs
        # self.predict_epochs = 1               # Number of prediction epochs
        # self.batch_size = 128                 # Batch size
        self.hid_dim = 256


