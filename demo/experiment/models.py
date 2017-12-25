import os
import torch
import torch.nn as nn
import torchvision.models as models
from spn.modules import SoftProposal, SpatialSumOverMap

class SPNetWSL(nn.Module):
    def __init__(self, model, num_classes, num_maps, pooling):
        super(SPNetWSL, self).__init__()

        self.features = nn.Sequential(*list(model.features.children())[:-1])
        self.spatial_pooling = pooling

        # classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_maps, num_classes)
        )

        # image normalization
        self.image_normalization_mean = [103.939, 116.779, 123.68]

    def forward(self, x): 
        x = self.features(x)
        x = self.spatial_pooling(x)
        x = x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def vgg16_sp(num_classes, pretrained=True, num_maps=1024):
    model = models.vgg16(pretrained=False)
    if pretrained:
        model_path = 'models/VGG16_ImageNet.pt'
        if os.path.isfile(model_path):
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
        else:
            print('Please download the pretrained VGG16 into ./models')

    num_features = model.features[28].out_channels
    pooling = nn.Sequential()
    pooling.add_module('adconv', nn.Conv2d(num_features, num_maps, kernel_size=3, stride=1, padding=1, groups=2, bias=True))
    pooling.add_module('maps', nn.ReLU())
    pooling.add_module('sp', SoftProposal())
    pooling.add_module('sum', SpatialSumOverMap())
    return SPNetWSL(model, num_classes, num_maps, pooling)

