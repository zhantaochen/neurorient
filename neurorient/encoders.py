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
        
        mask_S = min(input_shape[1:] if len(input_shape)==3 else input_shape)
        
        # # convolution 1
        # # first specify the output type of the convolutional layer
        # # we choose 24 feature fields, each transforming under the regular representation of C8
        # out_type = esnn.FieldType(self.r2_act, 12*[self.r2_act.regular_repr])
        # self.block1 = esnn.SequentialModule(
        #     esnn.MaskModule(in_type, mask_S, margin=1),
        #     esnn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
        #     esnn.InnerBatchNorm(out_type),
        #     esnn.ReLU(out_type, inplace=True)
        # )
        
        # # convolution 2
        # # the old output type is the input type to the next layer
        # in_type = self.block1.out_type
        # # the output type of the second convolution layer are 48 regular feature fields of C8
        # out_type = esnn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        # self.block2 = esnn.SequentialModule(
        #     esnn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
        #     esnn.InnerBatchNorm(out_type),
        #     esnn.ReLU(out_type, inplace=True)
        # )
        # self.pool1 = esnn.SequentialModule(
        #     esnn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        # )
        
        # # convolution 3
        # # the old output type is the input type to the next layer
        # in_type = self.block2.out_type
        # # the output type of the third convolution layer are 48 regular feature fields of C8
        # out_type = esnn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        # self.block3 = esnn.SequentialModule(
        #     esnn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
        #     esnn.InnerBatchNorm(out_type),
        #     esnn.ReLU(out_type, inplace=True)
        # )
        
        # # convolution 4
        # # the old output type is the input type to the next layer
        # in_type = self.block3.out_type
        # # the output type of the fourth convolution layer are 96 regular feature fields of C8
        # out_type = esnn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        # self.block4 = esnn.SequentialModule(
        #     esnn.R2Conv(in_type, out_type, kernel_size=3, padding=2, bias=False),
        #     esnn.InnerBatchNorm(out_type),
        #     esnn.ReLU(out_type, inplace=True)
        # )
        # self.pool2 = esnn.SequentialModule(
        #     esnn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        # )
        
        # # convolution 5
        # # the old output type is the input type to the next layer
        # in_type = self.block4.out_type
        # # the output type of the fifth convolution layer are 96 regular feature fields of C8
        # out_type = esnn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        # self.block5 = esnn.SequentialModule(
        #     esnn.R2Conv(in_type, out_type, kernel_size=3, padding=2, bias=False),
        #     esnn.InnerBatchNorm(out_type),
        #     esnn.ReLU(out_type, inplace=True)
        # )
        
        # # convolution 6
        # # the old output type is the input type to the next layer
        # in_type = self.block5.out_type
        # # the output type of the sixth convolution layer are 64 regular feature fields of C8
        # out_type = esnn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        # self.block6 = esnn.SequentialModule(
        #     esnn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
        #     esnn.InnerBatchNorm(out_type),
        #     esnn.ReLU(out_type, inplace=True)
        # )
        # self.pool3 = esnn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2, padding=0)

        # # convolution 7
        # # the old output type is the input type to the next layer
        # in_type = self.block6.out_type
        # # the output type of the fifth convolution layer are 96 regular feature fields of C8
        # out_type = esnn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        # self.block7 = esnn.SequentialModule(
        #     esnn.R2Conv(in_type, out_type, kernel_size=3, padding=2, bias=False),
        #     esnn.InnerBatchNorm(out_type),
        #     esnn.ReLU(out_type, inplace=True)
        # )
        
        # # convolution 8
        # # the old output type is the input type to the next layer
        # in_type = self.block7.out_type
        # # the output type of the sixth convolution layer are 64 regular feature fields of C8
        # out_type = esnn.FieldType(self.r2_act, 192*[self.r2_act.regular_repr])
        # self.block8 = esnn.SequentialModule(
        #     esnn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
        #     esnn.InnerBatchNorm(out_type),
        #     esnn.ReLU(out_type, inplace=True)
        # )
        # self.pool4 = esnn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2, padding=0)
        


        # Define lists for our parameters
        channels = [12, 24, 24, 48, 48, 96, 96, 192]
        kernel_sizes = [7, 5, 5, 3, 3, 3, 3, 3]
        paddings = [1, 2, 2, 2, 2, 1, 2, 1]
        pools_after_blocks = [2, 4, 6, 8]

        # For the first iteration, the input type is special
        # in_type = esnn.FieldType(self.r2_act, channels[0]*[self.r2_act.regular_repr])
        out_type = esnn.FieldType(self.r2_act, channels[0]*[self.r2_act.regular_repr])
        setattr(self, "block1", esnn.SequentialModule(
                    esnn.MaskModule(in_type, mask_S, margin=1),
                    esnn.R2Conv(in_type, out_type, kernel_size=kernel_sizes[0], padding=paddings[0], bias=False),
                    esnn.InnerBatchNorm(out_type),
                    esnn.ReLU(out_type, inplace=True)
                ))
        
        # For subsequent iterations
        for i in range(1, len(channels)):
            print(f"building block {i+1}")
            in_type = getattr(self, f"block{i}").out_type
            out_type = esnn.FieldType(self.r2_act, channels[i]*[self.r2_act.regular_repr])
            
            conv_module = esnn.SequentialModule(
                esnn.R2Conv(in_type, out_type, kernel_size=kernel_sizes[i], padding=paddings[i], bias=False),
                esnn.InnerBatchNorm(out_type),
                esnn.ReLU(out_type, inplace=True)
            )
            
            setattr(self, f"block{i+1}", conv_module)
            
            # Check if we should add a pooling layer after this block
            if (i+1) in pools_after_blocks:
                pool_module = esnn.SequentialModule(
                    esnn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
                )
                setattr(self, f"pool{(i+1)//2}", pool_module)


        self.gpool = esnn.GroupPooling(out_type)


    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = esnn.GeometricTensor(input, self.input_type)
        
        # apply each equivariant block
        
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)

        print(x.shape)
        
        self.feat_after_pool1 = x.tensor.clone().detach()

        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        print(x.shape)
        
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool3(x)
        print(x.shape)
        
        x = self.block7(x)
        x = self.block8(x)
        x = self.pool4(x)
        print(x.shape)

        # # pool over the group
        # x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor
        
        # classify with the final fully connected layers)
        # x = self.fully_net(x.reshape(x.shape[0], -1))
        
        return x