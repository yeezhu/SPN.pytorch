# functions/add.py
import torch
from torch.autograd import Function
from .utils import FunctionBackend
from .._ext import libspn


class SPGenerate(Function):
    def __init__(self, distanceMetric, transferMatrix, proposal, proposalBuffer, couple):
        super(SPGenerate, self).__init__()
        self.backend = FunctionBackend(libspn)
        self.distanceMetric = distanceMetric
        self.transferMatrix = transferMatrix
        self.proposal = proposal
        self.proposalBuffer = proposalBuffer
        self.couple = couple
        self.tolerance = 0.0001
        self.maxIteration = 20
        self.nBatch = 0
        self.mW = 0
        self.mH = 0

    def lazyInit(self, input):
        self.nBatch = input.size(0)
        self.mW = input.size(2)
        self.mH = input.size(3)
        self.N = self.mW * self.mH
        self.factor = 0.15 * self.N
        self.backend.SP_InitDistanceMetric(
            self.distanceMetric,
            self.factor,
            self.mW,
            self.mH,
            self.N)

    def forward(self, input):
        assert 'cuda' in input.type(), 'CPU version is currently not implemented'
        
        self.backend.set_type(input.type())

        if self.nBatch != input.size(0) or self.mW != input.size(2) or self.mH != input.size(3):
            self.lazyInit(input)

        output = input.new()
        self.backend.SP_Generate(
            input,
            self.distanceMetric,
            self.transferMatrix,
            self.proposal,
            self.proposalBuffer,
            self.tolerance,
            self.maxIteration)

        if self.couple:
            output.resize_(input.size())
            self.backend.SP_Couple(
                input,
                self.proposal,
                output)
        else:
            output.resize_(self.proposal.size())
            output = self.proposal.clone()

        self.save_for_backward(input)
        return output

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = input.new()
        
        if self.couple:
            grad_input.resize_(input.size())
            self.backend.SP_Couple(
                grad_output, 
                self.proposal, 
                grad_input)
        else:
            grad_input.resize_(input.size()).zero_()

        return grad_input