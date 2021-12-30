"""
Module having the building blocks for ResNet and one ResNet model
"""

# For defining an interface
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

# Building blocks
#===============================================================================

class ResNetBottleneck(nn.Module):
    """
    ResNet bottlneck

    This bottleneck does not change spatial dimensions, only the number of chanels (1x1 conv)
    Can be used for both downsamplig and upsampling the number of channels
    """

    def __init__(self, input_channels, output_channels):

        # Initialize parent class
        super().__init__()

        # Define the bottleneck
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bottleneck = nn.Conv2d(
            in_channels = self.input_channels,
            out_channels = self.output_channels,
            kernel_size = 1, # BottleNeck in depth, not spatially
            stride = 1       # BottleNeck in depth, not spatially
        )

    def forward(self, x):
        return F.relu(self.bottleneck(x))

class ResNetBlock(nn.Module):
    """
    Defines the interface for a ResNetBlock

    Multiple ResNetBlocks can be used for building the whole architecture
        - One can use simple blocks made of 2 conv layers
        - One can use more advance blocks, made of 1 bottleneck, 1 conv layer, 1 upsampling bottleneck

    So this defines what is a building block
    """

    # TODO -- write parameters in a way that do not produce errors
    def __init__(self):
        """
        Parameters:
        ===========
        input_channels: the number of channels of the input images
        number_of_filters: the number of channels of the output images
        kernel_size: size of the conv area
        stride: step size of the conv operation
        disable_identity: if True, F(x) is returned instead of F(x) + x
                          Useful when we go from one dimension to other between blocks
        """
        super().__init__()

    def forward(self, x: torch.Tensor):
        pass

class CommonBlock(ResNetBlock):
    """
    Basic building block of ResNet architecture
    Two convolutions that mantain spatial dimensions
    Zero padding is used in order to compute identity + x
    (tensors must have same dimensionalities)
    """

    def __init__(
        self,
        input_channels: int,
        number_of_filters: int,
        kernel_size: int = 3,
        stride: int = 1,
        disable_identity: bool = False, # Useful when we go from n filters to 2n filters
    ):
        # Init the parent
        super().__init__()


        # Saved parameters
        self.input_channels = input_channels
        self.number_of_filters = number_of_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.disable_identity = disable_identity

        # Operations
        self.conv1 = nn.Conv2d(
            in_channels = self.input_channels,
            out_channels = self.number_of_filters,
            kernel_size = self.kernel_size,
            stride = self.stride,
            padding = 1,
            padding_mode = "zeros"
        )

        self.conv2 = nn.Conv2d(
            in_channels = self.number_of_filters,
            out_channels = self.number_of_filters,
            kernel_size = self.kernel_size,
            stride = self.stride,
            padding = 1,
            padding_mode = "zeros"
        )

        # Batch normalizations
        self.batch_norm1 = nn.BatchNorm2d(num_features = self.number_of_filters)
        self.batch_norm2 = nn.BatchNorm2d(num_features = self.number_of_filters)


    def forward(self, x: torch.Tensor):
        # Save the identity of the input
        identity = x

        # First layer
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)

        # Second layer
        x = self.conv2(x)
        x = self.batch_norm2(x)

        # Apply residual formula
        if self.disable_identity is False:
            x = x + identity

        # Apply last non-linearity
        x = F.relu(x)

        return x

class BottleNeckBlock(ResNetBlock):
    """
    ResNet building block made of:
        - One bottleneck for reducing depth
        - One conv layer
        - One bottleneck for going to original depth

    This building block should be faster than CommonBlock due to the bottleneck applied before convolutions
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        disable_identity: bool = False, # Useful when we go from n filters to 2n filters
    ):
        # Init the parent class
        super().__init__()

        # Saved parameters
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.disable_identity = disable_identity

        # Operations

        # We are going to downsample to half channels of the input
        half_input_channels = int(self.input_channels / 2)

        # Security check of this parameter that can be 0
        if half_input_channels == 0: half_input_channels = 1

        # Reduce depth
        self.downsample = ResNetBottleneck(input_channels = self.input_channels, output_channels = half_input_channels)

        self.conv = nn.Conv2d(
            in_channels = half_input_channels,
            out_channels = half_input_channels,
            kernel_size = self.kernel_size,
            stride = self.stride,
            padding = 1,
            padding_mode = "zeros"
        )

        # Go to desired number of filters
        self.upsample = ResNetBottleneck(input_channels = half_input_channels, output_channels = self.output_channels)


        # Batch normalizations
        self.batch_norm1 = nn.BatchNorm2d(num_features = half_input_channels)
        self.batch_norm2 = nn.BatchNorm2d(num_features = half_input_channels)
        self.batch_norm3 = nn.BatchNorm2d(num_features = self.output_channels)


    def forward(self, x: torch.Tensor):
        # Save the identity of the input
        identity = x

        # Downsample
        x = self.downsample(x)
        x = self.batch_norm1(x)
        x = F.relu(x)

        # Apply convolution
        x = self.conv(x)
        x = self.batch_norm2(x)
        x = F.relu(x)

        # Apply upsampling
        x = self.upsample(x)
        x = self.batch_norm3(x)

        # Apply residual formula
        if self.disable_identity is False:
            x = x + identity

        # Apply last non-linearity
        x = F.relu(x)

        return x

# One model
#===============================================================================
class ResNetI(nn.Module):
    """Model based on simple ResNet"""

    def __init__(self, BuildingBlockClass: ResNetBlock):
        """
        Parameters:
        ============
        building_block: the basic ResNet building block class we are going to use to build the net
        """

        # Init the parent class
        super().__init__()

        # ResNet original architecture has an initial convolution
        self.init_conv = nn.Conv2d(
            in_channels = 3,
            out_channels = 64,
            kernel_size = 7,
            stride = 1,
            padding = 3,
            padding_mode = "zeros"
        )

        # ResNet blocks
        # We use a list of blocks because we are going
        # to have a lot of stacked blocks to work with
        self.resnet_blocks = []

        # First set of blocks
        # TODO -- BUG -- __init__() missing 2 required positional arguments: 'input_channels' and 'number_of_filters'
        for _ in range(3):
            self.resnet_blocks.append(BuildingBlockClass(
                input_channels = 64,
                number_of_filters = 64,
                kernel_size = 3,
                stride = 1,
                disable_identity = False, # Useful when we go from n filters to 2n filters
            ))


        # Second set of blocks
        for i in range(4):
            self.resnet_blocks.append(BuildingBlockClass(
                # if statement to do 64 to 128 channels transition
                input_channels = 64 if i == 0 else 128,
                number_of_filters =  128,
                kernel_size = 3,
                stride = 1,
                disable_identity = True if i == 0 else False
            ))

        # Third set of blocks
        for i in range(6):
            self.resnet_blocks.append(BuildingBlockClass(
                # if statement to do 64 to 128 channels transition
                input_channels = 128 if i == 0 else 256,
                number_of_filters =  256,
                kernel_size = 3,
                stride = 1,
                disable_identity = True if i == 0 else False

            ))

        # Fourth set of blocks
        for i in range(3):
            self.resnet_blocks.append(BuildingBlockClass(
                # if statement to do 64 to 128 channels transition
                input_channels = 256 if i == 0 else 512,
                number_of_filters =  512,
                kernel_size = 3,
                stride = 1,
                disable_identity = True if i == 0 else False
            ))


        # 512 channels with 32x32 images makes fc layer with too many parameters
        # We bottleneck the channels and also the image dimensions
        # TODO -- is max pooling a good idea? Because 32x32 is small enough
        self.bottleneck = ResNetBottleneck(512, 128)
        self.max_pooling = nn.MaxPool2d(2, 2)

        # Now only one fully connected layer for class scores
        self.fc = nn.Linear(32768, 100)

        # Pytorch needs to operate with their datastructures, not python lists
        # If we operate with gpu, python list of resnetblocks will be not in gpu mem
        # So this ModuleList acts as a list but with nn.Module objects
        self.resnet_blocks = nn.ModuleList(self.resnet_blocks)

    def forward(self, x):
        # Initial conv
        x = self.init_conv(x)

        # ResNet blocks
        for block in self.resnet_blocks:
            x = block(x)

        # Bottleneck and max-pooling
        x = self.bottleneck(x)
        x = self.max_pooling(x)

        # Fully-Connected layers for class-score
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
