import torch
from torch import nn
from otherlayers import *
import numpy as np
from torch_geometric.nn import GCNConv


class AMHMDA(nn.Module):
    def __init__(self, m_emd, d_emd, mdi):
        super(AMHMDA, self).__init__()
        self.Xm = m_emd
        self.Xd = d_emd
        self.md_supernode = mdi

    def forward(self, sim_data, train_data):##*
        Em = self.Xm(sim_data)
        Ed = self.Xd(sim_data)
        mFea, dFea = pro_data(train_data, Em, Ed)
        pre_asso = self.md_supernode(mFea, dFea)

        return pre_asso


def pro_data(data, em, ed):
    edgeData = data.t()

    mFeaData = em
    dFeaData = ed
    m_index = edgeData[0]
    d_index = edgeData[1]
    Em = torch.index_select(mFeaData, 0, m_index)
    Ed = torch.index_select(dFeaData, 0, d_index)

    return Em, Ed

# extract similarity feature
class EmbeddingM(nn.Module):
    def __init__(self, args):
        super(EmbeddingM, self).__init__()
        self.args = args

        self.gcn_x1_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_f = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x1_s = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_s = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x1_g = GCNConv(self.args.fm, self.args.fm)
        self.gcn_x2_g = GCNConv(self.args.fm, self.args.fm)

        self.fc1_x = nn.Linear(in_features=self.args.view * self.args.gcn_layers,
                               out_features=5 * self.args.view * self.args.gcn_layers)
        self.fc2_x = nn.Linear(in_features=5 * self.args.view * self.args.gcn_layers,
                               out_features=self.args.view * self.args.gcn_layers)
        self.sigmoidx = nn.Sigmoid()
        self.cnn_x = nn.Conv2d(in_channels=self.args.view * self.args.gcn_layers, out_channels=1,
                               kernel_size=(1, 1), stride=1, bias=True)

    def forward(self, data):
        torch.manual_seed(1)
        miRNA_number = len(data['mm_f']['data_matrix'])
        x_m = torch.randn(miRNA_number, self.args.fm)

        x_m_f1 = torch.relu(self.gcn_x1_f(x_m.cuda(), data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][
            data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))
        x_m_f2 = torch.relu(self.gcn_x2_f(x_m_f1, data['mm_f']['edges'].cuda(), data['mm_f']['data_matrix'][
            data['mm_f']['edges'][0], data['mm_f']['edges'][1]].cuda()))

        x_m_s1 = torch.relu(self.gcn_x1_s(x_m.cuda(), data['mm_s']['edges'].cuda(), data['mm_s']['data_matrix'][
            data['mm_s']['edges'][0], data['mm_s']['edges'][1]].cuda()))
        x_m_s2 = torch.relu(self.gcn_x2_s(x_m_s1, data['mm_s']['edges'].cuda(), data['mm_s']['data_matrix'][
            data['mm_s']['edges'][0], data['mm_s']['edges'][1]].cuda()))

        x_m_g1 = torch.relu(self.gcn_x1_g(x_m.cuda(), data['mm_g']['edges'].cuda(), data['mm_g']['data_matrix'][
            data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))
        x_m_g2 = torch.relu(self.gcn_x2_g(x_m_g1, data['mm_g']['edges'].cuda(), data['mm_g']['data_matrix'][
            data['mm_g']['edges'][0], data['mm_g']['edges'][1]].cuda()))

        XM = torch.cat((x_m_f1,x_m_f2,x_m_s1,x_m_s2,x_m_g1,x_m_g2), 1).t()
        XM = XM.view(1, self.args.view * self.args.gcn_layers, self.args.fm, -1)

        globalAvgPool_x = nn.AvgPool2d((self.args.fm, miRNA_number), (1, 1))
        x_channel_attention = globalAvgPool_x(XM)

        x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), -1)
        x_channel_attention = self.fc1_x(x_channel_attention)
        x_channel_attention = torch.relu(x_channel_attention)
        x_channel_attention = self.fc2_x(x_channel_attention)
        x_channel_attention = self.sigmoidx(x_channel_attention)
        x_channel_attention = x_channel_attention.view(x_channel_attention.size(0), x_channel_attention.size(1), 1, 1)
        XM_channel_attention = x_channel_attention * XM
        XM_channel_attention = torch.relu(XM_channel_attention)

        x = self.cnn_x(XM_channel_attention)
        x = x.view(self.args.fm, miRNA_number).t()

        return x


class EmbeddingD(nn.Module):
    def __init__(self, args):
        super(EmbeddingD, self).__init__()
        self.args = args

        self.gcn_y1_t = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_t = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y1_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_s = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y1_g = GCNConv(self.args.fd, self.args.fd)
        self.gcn_y2_g = GCNConv(self.args.fd, self.args.fd)

        self.fc1_y = nn.Linear(in_features=self.args.view * self.args.gcn_layers,
                               out_features=5 * self.args.view  * self.args.gcn_layers)
        self.fc2_y = nn.Linear(in_features=5 * self.args.view  * self.args.gcn_layers,
                               out_features=self.args.view  * self.args.gcn_layers)
        self.sigmoidy = nn.Sigmoid()
        self.cnn_y = nn.Conv2d(in_channels=self.args.view  * self.args.gcn_layers, out_channels=1,
                               kernel_size=(1, 1), stride=1, bias=True)

    def forward(self, data):
        torch.manual_seed(1)
        disease_number = len(data['dd_t']['data_matrix'])
        x_d = torch.randn(disease_number, self.args.fd)

        y_d_t1 = torch.relu(self.gcn_y1_t(x_d.cuda(), data['dd_t']['edges'].cuda(), data['dd_t']['data_matrix'][
            data['dd_t']['edges'][0], data['dd_t']['edges'][1]].cuda()))
        y_d_t2 = torch.relu(self.gcn_y2_t(y_d_t1, data['dd_t']['edges'].cuda(), data['dd_t']['data_matrix'][
            data['dd_t']['edges'][0], data['dd_t']['edges'][1]].cuda()))

        y_d_s1 = torch.relu(self.gcn_y1_s(x_d.cuda(), data['dd_s']['edges'].cuda(), data['dd_s']['data_matrix'][
            data['dd_s']['edges'][0], data['dd_s']['edges'][1]].cuda()))
        y_d_s2 = torch.relu(self.gcn_y2_s(y_d_s1, data['dd_s']['edges'].cuda(), data['dd_s']['data_matrix'][
            data['dd_s']['edges'][0], data['dd_s']['edges'][1]].cuda()))

        y_d_g1 = torch.relu(self.gcn_y1_g(x_d.cuda(), data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][
            data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))
        y_d_g2 = torch.relu(self.gcn_y2_g(y_d_g1, data['dd_g']['edges'].cuda(), data['dd_g']['data_matrix'][
            data['dd_g']['edges'][0], data['dd_g']['edges'][1]].cuda()))

        YD = torch.cat((y_d_t1,y_d_t2,y_d_s1,y_d_s2,y_d_g1,y_d_g2), 1).t()
        YD = YD.view(1, self.args.view  * self.args.gcn_layers, self.args.fd, -1)

        globalAvgPool_y = nn.AvgPool2d((self.args.fm, disease_number), (1, 1))
        y_channel_attention = globalAvgPool_y(YD)

        y_channel_attention = y_channel_attention.view(y_channel_attention.size(0), -1)
        y_channel_attention = self.fc1_y(y_channel_attention)
        y_channel_attention = torch.relu(y_channel_attention)
        y_channel_attention = self.fc2_y(y_channel_attention)
        y_channel_attention = self.sigmoidy(y_channel_attention)
        y_channel_attention = y_channel_attention.view(y_channel_attention.size(0), y_channel_attention.size(1), 1, 1)

        YD_channel_attention = y_channel_attention * YD
        YD_channel_attention = torch.relu(YD_channel_attention)

        y = self.cnn_y(YD_channel_attention)
        y = y.view(self.args.fd, disease_number).t()

        return y



#construct the hyper-graph
class MDI(nn.Module):
    def __init__(self, param):
        super(MDI, self).__init__()

        self.inSize = param.inSize
        self.outSize = param.outSize
        self.gcnlayers = param.gcn_layers
        self.device = param.device
        self.nodeNum = param.nodeNum
        self.hdnDropout = param.hdnDropout
        self.fcDropout = param.fcDropout
        self.maskMDI = param.maskMDI
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.LeakyReLU()

        self.nodeEmbedding = BnodeEmbedding(
            torch.tensor(np.random.normal(size=(max(self.nodeNum, 0), self.inSize)), dtype=torch.float32),
            dropout=self.hdnDropout).to(self.device)

        self.nodeGCN = GCN(self.inSize, self.outSize, dropout=self.hdnDropout, layers=self.gcnlayers, resnet=True,
                           actFunc=self.relu1).to(self.device)

        self.fcLinear = MLP(self.outSize, 1, dropout=self.fcDropout, actFunc=self.relu1).to(self.device)

        self.layeratt_m = LayerAtt(self.inSize, self.outSize, self.gcnlayers)
        self.layeratt_d = LayerAtt(self.inSize, self.outSize, self.gcnlayers)

    def forward(self, em, ed):
        xm = em.unsqueeze(1)
        xd = ed.unsqueeze(1)
        if self.nodeNum > 0:
            node = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(self.nodeEmbedding.embedding.weight)).repeat(
                len(xd), 1, 1)
            node = torch.cat([xm, xd, node], dim=1)
            nodeDist = torch.sqrt(torch.sum(node ** 2, dim=2, keepdim=True) + 1e-8)
            cosNode = torch.matmul(node, node.transpose(1, 2)) / (nodeDist * nodeDist.transpose(1, 2) + 1e-8)
            # cosNode = cosNode*0.5 + 0.5
            cosNode = self.relu2(cosNode)
            cosNode[:, range(node.shape[1]), range(node.shape[1])] = 1
            if self.maskMDI: cosNode[:, 0, 1] = cosNode[:, 1, 0] = 0
            D = torch.eye(node.shape[1], dtype=torch.float32, device=self.device).repeat(len(xm), 1, 1)
            D[:, range(node.shape[1]), range(node.shape[1])] = 1 / (torch.sum(cosNode, dim=2) ** 0.5)
            pL = torch.matmul(torch.matmul(D, cosNode), D)

            mGCNem, dGCNem = self.nodeGCN(node, pL)
            mLAem = self.layeratt_m(mGCNem)
            dLAem = self.layeratt_d(dGCNem)
            node_embed = mLAem * dLAem

        else:
            node_embed = (xm * xd).squeeze(dim=1)
        pre_part = self.fcLinear(node_embed)
        pre_a = self.sigmoid(pre_part).squeeze(dim=1)

        return pre_a
