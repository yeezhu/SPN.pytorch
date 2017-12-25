from .SPG import SPGenerate
from .SSO import SpatialSumOverMapFunc

def sp_generate(input, distanceMetric, transferMatrix, proposal, proposalBuffer, factor, couple=True):
  return SPGenerate(distanceMetric, transferMatrix, proposal, proposalBuffer, factor, couple)(input)

def spatial_sum_over_map(input):
  return SpatialSumOverMapFunc()(input)
  
__all__ = ['SPGenerate', 'sp_generate', 'SpatialSumOverMapFunc', 'spatial_sum_over_map']