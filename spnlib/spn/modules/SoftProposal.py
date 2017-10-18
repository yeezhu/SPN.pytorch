import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..functions import sp_generate

class SoftProposal(nn.Module):
    def __init__(self, couple=True, factor=0.15):
        super(SoftProposal, self).__init__()
        self.couple = couple   
        self.factor = factor
        self.nBatch = 0
        self.mW = 0
        self.mH = 0

    def lazyInit(self, input):
        self.nBatch = input.size(0)
        self.mW = input.size(2)
        self.mH = input.size(3)
        self.N = self.mW * self.mH

        input_data = input.data
        self.distanceMetric = input_data.new()
        self.distanceMetric.resize_(self.N, self.N)
        self.transferMatrix = input_data.new()
        self.transferMatrix.resize_(self.N, self.N)
        self.proposal = input_data.new()
        self.proposal.resize_(self.nBatch, self.mW, self.mH)
        self.proposalBuffer = input_data.new()
        self.proposalBuffer.resize_(self.mW, self.mH)

    def forward(self, input):
        if self.nBatch != input.size(0) or self.mW != input.size(2) or self.mH != input.size(3):
            self.lazyInit(input)

        return sp_generate(input, self.distanceMetric, self.transferMatrix, self.proposal, self.proposalBuffer, self.couple)

    def __repr__(self):
        sp_config = '[couple={},factor={}]'.format(self.couple, self.factor)
        s = ('{name}({sp_config})')
        return s.format(name=self.__class__.__name__, sp_config=sp_config, **self.__dict__)
