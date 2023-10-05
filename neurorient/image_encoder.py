"""
Use pytorch's ResNet as the image encoder.

- Adjust the channel in the first layer from 3 to 1.
- Save intermediate output for downstream BiFPN feature fusion.

The link to the pytorch's implementation:
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""

import torch
import torch.nn as nn

import torchvision.models.resnet as resnet

class ImageEncoder(nn.Module):

    def __init__(self, backbone_type = "resnet18", pretrained = True):
        super().__init__()

        self.backbone_dict = {
            "resnet18" : resnet.resnet18,
            "resnet50" : resnet.resnet50,
        }
        assert backbone_type in self.backbone_dict, f"{backbone_type} is not supported!!!"
        if backbone_type.find("resnet") > -1:
            self.backbone = self.backbone_dict[backbone_type](weights='DEFAULT' if pretrained else None)
        else:
            raise NotImplementedError(f"{backbone_type} is not supported!!!")

        self.adjust_layers()

        self.save_layer = {
            "relu"   : True,
            "layer1" : True,
            "layer2" : True,
            "layer3" : True,
            "layer4" : True,
        }

        self.output_channels = self.get_output_channels(backbone_type)


    def get_output_channels(self, backbone_type):
        return {
            'resnet18' : {
                "relu"   : 64,
                "layer1" : 64,
                "layer2" : 128,
                "layer3" : 256,
                "layer4" : 512,
            },
            'resnet50' : {
                "relu"   : 64,
                "layer1" : 256,
                "layer2" : 512,
                "layer3" : 1024,
                "layer4" : 2048,
            }
        }[backbone_type]


    def adjust_layers(self):
        # Average the weights in the input channels...
        conv1_weight = self.backbone.conv1.weight.data.mean(dim = 1, keepdim = True)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.backbone.conv1.weight.data = conv1_weight

        # Ignore the avgpool and fc layer
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc      = nn.Identity()


    def forward(self, x):
        fmap_in_layers = []

        # Going through the immediate layers and collect feature maps...
        for name, backbone_layer in self.backbone.named_children():
            x = backbone_layer(x)

            # Save fmap from all layers except these excluded...
            if self.save_layer.get(name, False):
                fmap_in_layers.append(x)

        ret = fmap_in_layers if len(fmap_in_layers) > 0 else x
        return ret
