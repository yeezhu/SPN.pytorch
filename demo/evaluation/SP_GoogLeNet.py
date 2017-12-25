import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Function, Variable
from torch.utils.serialization import load_lua
from spn import SoftProposal, SpatialSumOverMap, hook_spn, unhook_spn


class CallLegacyModel(Function):
    @staticmethod
    def forward(ctx, model, x):
        if x.is_cuda:
            return model.cuda().forward(x)
        else:
            return model.float().forward(x)
        
    @staticmethod
    def backward(ctx, *args, **kwargs):
        raise NotImplementedError('The backward call of LegacyModel is not implemented')

class LegacyModel(nn.Module):
    def __init__(self, model):
        super(LegacyModel, self).__init__()
        self.model = model
        
    def forward(self, x):
        return CallLegacyModel.apply(self.model, x)
    
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, repr(self.model))

class SP_GoogLeNet(nn.Module):
    def __init__(self, state_dict='SP_GoogleNet_ImageNet.pt'):
        super(SP_GoogLeNet, self).__init__()

        state_dict = load_lua(state_dict)
        pretrained_model = state_dict[0]
        pretrained_model.evaluate()
        
        self.features = LegacyModel(pretrained_model)
        self.pooling = nn.Sequential()
        self.pooling.add_module('adconv', nn.Conv2d(832, 1024, kernel_size=3, stride=1, padding=1, groups=2, bias=True))
        self.pooling.add_module('maps', nn.ReLU())
        self.pooling.add_module('sp', SoftProposal(factor=2.1))
        self.pooling.add_module('sum', SpatialSumOverMap())

        self.pooling.adconv.weight.data.copy_(state_dict[1][0])
        self.pooling.adconv.bias.data.copy_(state_dict[1][1])

        # classification layer
        self.classifier = nn.Linear(1024, 1000)

        self.classifier.weight.data.copy_(state_dict[2][0])
        self.classifier.bias.data.copy_(state_dict[2][1])

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, x): 
        x = self.features(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def inference(self, mode=True):
        hook_spn(self) if mode else unhook_spn(self)
        return self
