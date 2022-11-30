from torch import nn as nn
import torch
from math import sqrt


class BatchNorm1d(nn.Module):
    def __init__(self, inSize, name='batchNorm1d'):
        super(BatchNorm1d, self).__init__()
        self.bn = nn.BatchNorm1d(inSize)
        self.name = name

    def forward(self, x):
        return self.bn(x)


class BnodeEmbedding(nn.Module):
    def __init__(self, embedding, dropout, freeze=False):
        super(BnodeEmbedding, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.as_tensor(embedding, dtype=torch.float32).detach(), freeze=freeze)
        self.dropout1 = nn.Dropout2d(p=dropout / 2)
        self.dropout2 = nn.Dropout(p=dropout / 2)
        self.p = dropout

    def forward(self, x):

        if self.p > 0:
            x = self.dropout2(self.dropout1(self.embedding(x)))
        else:
            x = self.embedding(x)
        return x


class MLP(nn.Module):
    def __init__(self, inSize, outSize, dropout, actFunc, outBn=True, outAct=False, outDp=False):
        super(MLP, self).__init__()
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(outSize)
        self.out = nn.Linear(inSize, outSize)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp

    def forward(self, x):
        x = self.out(x)#batchsize*featuresize
        if self.outBn: x = self.bns(x) if len(x.shape) == 2 else self.bns(x.transpose(-1, -2)).transpose(-1, -2)
        if self.outAct: x = self.actFunc(x)
        if self.outDp: x = self.dropout(x)
        return x


class GCN(nn.Module):
    def __init__(self, inSize, outSize, dropout, layers, resnet, actFunc, outBn=False, outAct=True, outDp=True):
        super(GCN, self).__init__()
        self.gcnlayers = layers
        self.actFunc = actFunc
        self.dropout = nn.Dropout(p=dropout)
        self.bns = nn.BatchNorm1d(outSize)
        self.out = nn.Linear(inSize, outSize)
        self.outBn = outBn
        self.outAct = outAct
        self.outDp = outDp
        self.resnet = resnet

    def forward(self, x, L):
        Z_zero = x# batchsize*node_num*featuresize
        m_all = Z_zero[:, 0, :].unsqueeze(dim=1)#batchsize*1*featuesize
        d_all = Z_zero[:, 1, :].unsqueeze(dim=1)

        for i in range(self.gcnlayers):
            a = self.out(torch.matmul(L, x))
            if self.outBn:
                if len(L.shape) == 3:
                    a = self.bns(a.transpose(1, 2)).transpose(1, 2)
                else:
                    a = self.bns(a)
            if self.outAct: a = self.actFunc(a)
            if self.outDp: a = self.dropout(a)
            if self.resnet and a.shape == x.shape:
                a += x
            x = a
            m_this = x[:, 0, :].unsqueeze(dim=1)
            d_this = x[:, 1, :].unsqueeze(dim=1)
            m_all = torch.cat((m_all, m_this), 1)
            d_all = torch.cat((d_all, d_this), 1)


        return m_all, d_all



class LayerAtt(nn.Module):
    def __init__(self, inSize, outSize, gcnlayers):
        super(LayerAtt, self).__init__()
        self.layers = gcnlayers + 1
        self.inSize = inSize
        self.outSize = outSize
        self.q = nn.Linear(inSize, outSize)
        self.k = nn.Linear(inSize, outSize)
        self.v = nn.Linear(inSize, outSize)
        self.norm = 1 / sqrt(outSize)
        self.actfun1 = nn.Softmax(dim=1)
        self.actfun2 = nn.ReLU()
        self.attcnn = nn.Conv1d(in_channels=self.layers, out_channels=1, kernel_size=1, stride=1,
                            bias=True)

    def forward(self, x):# batchsize*gcn_layers*featuresize
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        out = torch.bmm(Q, K.permute(0, 2, 1)) * self.norm
        alpha = self.actfun1(out)# according to gcn_layers
        z = torch.bmm(alpha, V)
        # cnnz = self.actfun2(z)
        cnnz = self.attcnn(z)
        # cnnz = self.actfun2(cnnz)
        finalz = cnnz.squeeze(dim=1)

        return finalz
