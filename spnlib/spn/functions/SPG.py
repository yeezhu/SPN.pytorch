# functions/add.py
import re
import torch
from torch.autograd import Function
from .._ext import libspn

class FunctionBackend(object):

    def __init__(self, lib):
        self.backends = dict()
        self.parse_lib(lib)
        self.current_backend = None

    def __getattr__(self, name):
        func = self.backends[self.current_backend].get(name)
        if func is None:
            raise NotImplementedError(name)
        return func

    def set_type(self, input_type):
        if input_type != self.current_backend:
            if not input_type in self.backends.keys():
                raise NotImplementedError("{} is not supported".format(input_type))
            self.current_backend = input_type

    def parse_lib(self, lib):
        for func in dir(lib):
            if func.startswith('_'):
                continue
            match_obj = re.match(r"(\w+)_(Float|Double)_(.+)", func)
            if match_obj:
                if match_obj.group(1).startswith("cu"):
                    backend = "torch.cuda.{}Tensor".format(match_obj.group(2))
                else:
                    backend = "torch.{}Tensor".format(match_obj.group(2))
                func_name = match_obj.group(3)
                if backend not in self.backends.keys():
                    self.backends[backend] = dict()
                self.backends[backend][func_name] = getattr(lib, func)

class SPGenerate(Function):
    def __init__(self, distanceMetric, transferMatrix, proposal, proposalBuffer, factor, couple):
        super(SPGenerate, self).__init__()
        self.backend = FunctionBackend(libspn)
        self.distanceMetric = distanceMetric
        self.transferMatrix = transferMatrix
        self.proposal = proposal
        self.proposalBuffer = proposalBuffer
        self.factor = factor
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