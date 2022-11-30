import numpy as np
from scipy.sparse import coo_matrix
import os
import torch
import csv
import torch.utils.data.dataset as Dataset


def dense2sparse(matrix: np.ndarray):
    mat_coo = coo_matrix(matrix)
    edge_idx = np.vstack((mat_coo.row, mat_coo.col))
    return edge_idx, mat_coo.data



def loading_data(param):
    ratio = param.ratio
    md_matrix = np.loadtxt(os.path.join(param.datapath+'/m_d.csv'), dtype=int, delimiter=',')

    # get the edge of positives samples
    rng = np.random.default_rng(seed=42)  # 固定训练测试
    pos_samples = np.where(md_matrix == 1)
    pos_samples_shuffled = rng.permutation(pos_samples, axis=1)

    # get the edge of negative samples
    rng = np.random.default_rng(seed=42)
    neg_samples = np.where(md_matrix == 0)
    neg_samples_shuffled = rng.permutation(neg_samples, axis=1)[:, :pos_samples_shuffled.shape[1]]

    edge_idx_dict = dict()
    n_pos_samples = pos_samples_shuffled.shape[1]

    idx_split = int(n_pos_samples * ratio)

    ##seed=42. The data classes in test and train are the same as those in './train_test/'
    test_pos_edges = pos_samples_shuffled[:, :idx_split]
    test_neg_edges = neg_samples_shuffled[:, :idx_split]
    test_pos_edges = test_pos_edges.T
    test_neg_edges = test_neg_edges.T
    test_true_label = np.hstack((np.ones(test_pos_edges.shape[0]), np.zeros(test_neg_edges.shape[0])))
    test_true_label = np.array(test_true_label, dtype='float32')
    test_edges = np.vstack((test_pos_edges, test_neg_edges))
    # np.savetxt('./train_test/test_pos.csv', test_pos_edges, delimiter=',')
    # np.savetxt('./train_test/test_neg.csv', test_neg_edges, delimiter=',')

    train_pos_edges = pos_samples_shuffled[:, idx_split:]
    train_neg_edges = neg_samples_shuffled[:, idx_split:]
    train_pos_edges = train_pos_edges.T
    train_neg_edges = train_neg_edges.T
    train_true_label = np.hstack((np.ones(train_pos_edges.shape[0]), np.zeros(train_neg_edges.shape[0])))
    train_true_label = np.array(train_true_label, dtype='float32')
    train_edges = np.vstack((train_pos_edges, train_neg_edges))
    # np.savetxt('./train_test/train_pos.csv', train_pos_edges, delimiter=',')
    # np.savetxt('./train_test/train_neg.csv', train_neg_edges, delimiter=',')

    edge_idx_dict['train_Edges'] = train_edges
    edge_idx_dict['train_Labels'] = train_true_label

    edge_idx_dict['test_Edges'] = test_edges
    edge_idx_dict['test_Labels'] = test_true_label

    edge_idx_dict['true_md'] = md_matrix##*

    return edge_idx_dict



def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(md_data)


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.LongTensor(edge_index)


def Simdata_pro(param):
    dataset = dict()

    "miRNA sequence sim"
    mm_s_matrix = read_csv(param.datapath + '/m_ss.csv')
    mm_s_edge_index = get_edge_index(mm_s_matrix)
    dataset['mm_s'] = {'data_matrix': mm_s_matrix, 'edges': mm_s_edge_index}

    "disease target-based sim"
    dd_t_matrix = read_csv(param.datapath + '/d_ts.csv')
    dd_t_edge_index = get_edge_index(dd_t_matrix)
    dataset['dd_t'] = {'data_matrix': dd_t_matrix, 'edges': dd_t_edge_index}

    "miRNA functional sim"
    mm_f_matrix = read_csv(param.datapath + '/m_fs.csv')
    mm_f_edge_index = get_edge_index(mm_f_matrix)
    dataset['mm_f'] = {'data_matrix': mm_f_matrix, 'edges': mm_f_edge_index}

    "disease semantic sim"
    dd_s_matrix = read_csv(param.datapath + '/d_ss.csv')
    dd_s_edge_index = get_edge_index(dd_s_matrix)
    dataset['dd_s'] = {'data_matrix': dd_s_matrix, 'edges': dd_s_edge_index}

    "miRNA Gaussian sim"
    mm_g_matrix = read_csv(param.datapath + '/m_gs.csv')
    mm_g_edge_index = get_edge_index(mm_g_matrix)
    dataset['mm_g'] = {'data_matrix': mm_g_matrix, 'edges': mm_g_edge_index}

    "disease Gaussian sim"
    dd_g_matrix = read_csv(param.datapath + '/d_gs.csv')
    dd_g_edge_index = get_edge_index(dd_g_matrix)
    dataset['dd_g'] = {'data_matrix': dd_g_matrix, 'edges': dd_g_edge_index}

    return dataset


class CVEdgeDataset(Dataset.Dataset):
    def __init__(self, edges, labels):

        self.Data = edges
        self.Label = labels

    def __len__(self):
        return len(self.Label)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label




