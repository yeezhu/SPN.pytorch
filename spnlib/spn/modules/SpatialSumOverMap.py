import torch
from torch.nn import Module
from ..functions import spatial_sum_over_map

class SpatialSumOverMap(Module):
    def __init__(self):
        super(SpatialSumOverMap, self).__init__()
    
    def forward(self, input):
        return spatial_sum_over_map(input)

    def __repr__(self):
        return self.__class__.__name__