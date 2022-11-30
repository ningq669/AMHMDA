from torch import nn as nn
import torch


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
        # x: batchSize × seqLen
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
        x = self.out(x)
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

    def forward(self, x, L):# x: nodeNum × feaSize; L: batchSize × nodeNum × nodeNum
        for i in range(self.gcnlayers):
            a = self.out(torch.matmul(L, x))  # => batchSize × nodeNum × outSize
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

        return x


class LayerNormAndDropout(nn.Module):
    def __init__(self, feaSize, dropout=0.1, name='layerNormAndDropout'):
        super(LayerNormAndDropout, self).__init__()
        self.layerNorm = nn.LayerNorm(feaSize)
        self.dropout = nn.Dropout(p=dropout)
        self.name = name

    def forward(self, x):
        return self.dropout(self.layerNorm(x))



