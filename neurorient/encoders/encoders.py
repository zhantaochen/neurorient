import torch
from torch import nn
import torchvision

from escnn import gspaces
from escnn import nn as esnn

class ImageEncoder(nn.Module):
    '''Define an image encoding network to process image into dense feature map

    Any standard convolutional network or vision transformer could be used here. 
    In the paper, we use ResNet50 pretrained on ImageNet1K for a fair comparison to
    the baselines.  Here, we show an example using a pretrained SWIN Transformer.

    When using a model from torchvision, make sure to remove the head so the output
    is a feature map, not a feature vector
    '''
    def __init__(self):
        super().__init__()
        self.layers = torchvision.models.swin_v2_t(weights="DEFAULT")

        # last three modules in swin are avgpool,flatten,linear so change to Identity
        self.layers.avgpool = nn.Identity()
        self.layers.flatten = nn.Identity()
        self.layers.head = nn.Identity()

        # we will need shape of feature map for later
        with torch.no_grad():
            dummy_input = torch.zeros((1, 3, 128, 128))
            self.output_shape = self(dummy_input).shape[1:]

    def forward(self, x):
        return self.layers(x)

class SteerableCNN(torch.nn.Module):
    
    def __init__(self, 
                 input_shape: tuple,
                 N: int=8
                ):
        
        super().__init__()
        
        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.rot2dOnR2(N)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = esnn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        self.input_shape = input_shape
        self.mask_S = min(self.input_shape[1:] if len(self.input_shape)==3 else self.input_shape)


        # Define lists for our parameters
        channels = [12, 24, 24, 48, 48, 96, 96, 192]
        kernel_sizes = [7, 5, 5, 3, 3, 3, 3, 3]
        paddings = [3, 2, 2, 1, 1, 1, 1, 1]
        pools_after_blocks = [2, 4, 6, 8]
        
        self.build_blocks(channels, kernel_sizes, paddings, pools_after_blocks)
        self.gpool = esnn.GroupPooling(self.blocks[-1].out_type)
        self.get_block_output_shapes()

    def build_blocks(
        self, 
        channels: tuple,
        kernel_sizes: tuple,
        paddings: tuple,
        pools_after_blocks: tuple
        ):
        self.blocks = nn.ModuleList()

        out_type = esnn.FieldType(self.r2_act, channels[0]*[self.r2_act.regular_repr])
        self.blocks.append(esnn.SequentialModule(
                esnn.MaskModule(self.input_type, self.mask_S, margin=1),
                esnn.R2Conv(self.input_type, out_type, kernel_size=kernel_sizes[0], padding=paddings[0], bias=False),
                esnn.InnerBatchNorm(out_type),
                esnn.ReLU(out_type, inplace=True)
            )
        )
        
        # For subsequent iterations
        for i in range(1, len(channels)):
            in_type = self.blocks[-1].out_type
            out_type = esnn.FieldType(self.r2_act, channels[i]*[self.r2_act.regular_repr])
            self.blocks.append(esnn.SequentialModule(
                    esnn.R2Conv(in_type, out_type, kernel_size=kernel_sizes[i], padding=paddings[i], bias=False),
                    esnn.InnerBatchNorm(out_type),
                    esnn.ReLU(out_type, inplace=True)
                )
            )
            # Check if we should add a pooling layer after this block
            if (i+1) in pools_after_blocks:
                self.blocks.append(
                    esnn.SequentialModule(esnn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2))
                    )

    def get_block_output_shapes(self,):
        self.block_output_shapes = []
        x = torch.zeros((1,) + self.input_shape)
        x = esnn.GeometricTensor(x, self.input_type)
        with torch.no_grad():
            for i_module, _module in enumerate(self.blocks):
                x = _module(x)
                self.block_output_shapes.append(list(x.shape[-2:]))

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = esnn.GeometricTensor(input, self.input_type)
        
        # apply each equivariant block
        for i_module, _module in enumerate(self.blocks):
            x = _module(x)
            if i_module == 2:
                self.feat_after_pool1 = x.tensor.clone()

        # # pool over the group
        # x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor
        return x