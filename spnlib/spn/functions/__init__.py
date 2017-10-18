from .SPG import SPGenerate

def sp_generate(input, distanceMetric, transferMatrix, proposal, proposalBuffer, couple=True):
  return SPGenerate(distanceMetric, transferMatrix, proposal, proposalBuffer, couple)(input)
  
__all__ = ["sp_generate"]