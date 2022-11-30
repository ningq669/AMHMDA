import torch
import random
import numpy as np
from datapro import Simdata_pro,loading_data

from train import train_test
# import warnings
#
# warnings.filterwarnings("ignore")


class Config:
    def __init__(self):
        self.datapath = './datasets'
        self.kfold = 5
        self.batchSize = 128
        self.ratio = 0.2
        self.epoch = 8
        self.gcn_layers = 2
        self.view = 3
        self.fm = 128
        self.fd = 128
        self.inSize = 128
        self.outSize = 128
        self.nodeNum = 64
        self.hdnDropout = 0.5
        self.fcDropout = 0.5
        self.maskMDI = False
        self.device = torch.device('cuda')



def main():
    param = Config()
    simData = Simdata_pro(param)
    train_data = loading_data(param)
    result = train_test(simData, train_data, param, state='valid')


if __name__ == "__main__":
    main()
