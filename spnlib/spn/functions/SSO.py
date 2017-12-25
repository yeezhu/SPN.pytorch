import torch
import torch.nn as nn
from torch.autograd import Function, Variable 


class SpatialSumOverMapFunc(Function):
    def __init__(self):
        super(SpatialSumOverMapFunc, self).__init__()

    def forward(self, input):
        batch_size, num_channels, h, w = input.size()
        x = input.view(batch_size, num_channels, h*w)
        output = torch.sum(x, 2)
        self.save_for_backward(input)
        return output.view(batch_size, num_channels)

    def backward(self, grad_output):
        input, = self.saved_tensors
        batch_size, num_channels, h, w = input.size()
        grad_input = grad_output.view(batch_size, num_channels, 1, 1).expand(batch_size, num_channels, h, w).contiguous()
        return grad_input