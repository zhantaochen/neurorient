import torch
import torch.nn            as nn
import torch.nn.functional as F

from .config import bifpn_internal_config




def conv2d(in_channels, out_channels, kernel_size, *, stride = 1, groups = 1, bias = False):    # ...`*` forces the rest arguments to be keyword arguments
    """Helper for building a conv2d layer."""
    assert kernel_size % 2 == 1, "Only odd size kernels supported to avoid padding issues."

    padding = (kernel_size - 1)//2

    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size = kernel_size,
                     stride      = stride,
                     padding     = padding,
                     groups      = groups,
                     bias        = bias)


def pool2d(kernel_size, *, stride = 2):    # ...`*` forces the rest arguments to be keyword arguments
    """Helper for building a pool2d layer."""
    assert kernel_size % 2 == 1, "Only odd size kernels supported to avoid padding issues."

    padding = (kernel_size - 1)//2

    return nn.MaxPool2d(kernel_size = kernel_size, stride = stride, padding = padding)




class DepthwiseSeparableConv2d(nn.Module):
    """
    As the name suggests, it's a conv2d doen in two steps:
    - Spatial only conv, no inter-channel communication.
    - Inter-channel communication, no spatial communication.
    """

    def __init__(self, in_channels,
                       out_channels,
                       kernel_size  = 1,
                       stride       = 1,
                       padding      = 0,
                       dilation     = 1,
                       groups       = 1,
                       bias         = True,
                       padding_mode = 'zeros',
                       device       = None,
                       dtype        = None):
        super().__init__()

        # Depthwise conv means channels are independent, only spatial bits communicate
        # Essentially it simply scales every tensor element
        self.depthwise_conv = nn.Conv2d(in_channels  = in_channels,
                                        out_channels = in_channels,
                                        kernel_size  = kernel_size,
                                        stride       = stride,
                                        padding      = padding,
                                        dilation     = dilation,
                                        groups       = in_channels,    # Input channels don't talk to each other
                                        bias         = bias,
                                        padding_mode = padding_mode,
                                        device       = device,
                                        dtype        = dtype)

        # Pointwise to facilitate inter-channel communication, no spatial bits communicate
        self.pointwise_conv = nn.Conv2d(in_channels  = in_channels,
                                        out_channels = out_channels,
                                        kernel_size  = 1,
                                        stride       = 1,
                                        padding      = 0,
                                        dilation     = 1,
                                        groups       = 1,    # Input channels don't talk to each other
                                        bias         = bias,
                                        padding_mode = padding_mode,
                                        device       = device,
                                        dtype        = dtype)


    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        return x




class BiFPNBlock(nn.Module):
    """
    One BiFPN block takes feature maps at five different scales:
    (p3, p4, ..., pN).

    Notice that the input to BiFPNBlock should have the same channel size, 
    which can be achieved with a conv layer in the upstream.
    """

    def __init__(self, num_features = 64, num_levels = 5):
        super().__init__()

        # Follow the paper's notation with base level starting from "3"...
        BASE_LEVEL = 3

        # Confusingly, there should be at least 3 levels in total...
        num_levels = max(num_levels, 3)

        # Decide the min max level...
        min_level  = BASE_LEVEL
        max_level  = BASE_LEVEL + (num_levels - 1)

        # Create conv2d layers for fusion stage M...
        # min_level, ..., max_level - 1
        m_conv = nn.ModuleDict({
            f"m{level}" : nn.Sequential(
                DepthwiseSeparableConv2d(in_channels  = num_features,
                                         out_channels = num_features,
                                         bias         = False),
                nn.BatchNorm2d(num_features = num_features,
                               eps          = bifpn_internal_config.MODEL.BIFPN.BN.EPS,
                               momentum     = bifpn_internal_config.MODEL.BIFPN.BN.MOMENTUM),
                nn.ReLU(),
            )
            for level in range(min_level, max_level)
        })

        # Create conv2d layers for fusion stage Q...
        # min_level + 1, max_level
        q_conv = nn.ModuleDict({
            f"q{level}" : nn.Sequential(
                DepthwiseSeparableConv2d(in_channels  = num_features,
                                         out_channels = num_features,
                                         bias         = False),
                nn.BatchNorm2d(num_features = num_features,
                               eps          = bifpn_internal_config.MODEL.BIFPN.BN.EPS,
                               momentum     = bifpn_internal_config.MODEL.BIFPN.BN.MOMENTUM),
                nn.ReLU(),
            )
            for level in range(min_level + 1, max_level + 1)
        })

        self.conv = nn.ModuleDict()
        self.conv.update(m_conv)
        self.conv.update(q_conv)

        # Define the weights used in fusion
        num_level_stage_m = max_level - min_level
        num_level_stage_q = num_level_stage_m
        self.w_m = nn.Parameter(torch.randn(num_level_stage_m, 2))    # Two-component fusion at stage M
        self.w_q = nn.Parameter(torch.randn(num_level_stage_q, 3))    # Three-component fusion at stage Q

        # Keep these numbers as attributes...
        self.BASE_LEVEL = BASE_LEVEL
        self.min_level  = min_level
        self.max_level  = max_level


    def forward(self, x):
        # Keep these numbers as attributes...
        BASE_LEVEL = self.BASE_LEVEL
        min_level  = self.min_level
        max_level  = self.max_level
        num_levels = max_level - min_level + 1

        # Unpack feature maps into dict...
        # x is 0-based index
        # (B, C, [H], [W])
        p = { level : x[idx] for idx, level in enumerate(range(min_level, min_level + num_levels)) }

        # ___/ Stage M \___
        # Fuse features from low resolution to high resolution (pathway M)...
        m = {}
        for idx, level_low in enumerate(range(max_level, min_level, -1)):
            level_high = level_low - 1
            m_low   = p[level_low ] if idx == 0 else m[level_low]
            p_high  = p[level_high]

            w1, w2 = self.w_m[idx]
            m_low_up = F.interpolate(m_low,
                                     scale_factor  = bifpn_internal_config.MODEL.BIFPN.UP_SCALE_FACTOR,
                                     mode          = 'bilinear',
                                     align_corners = False)
            m_fused  = w1 * p_high + w2 * m_low_up
            m_fused /= (w1 + w2 + bifpn_internal_config.MODEL.BIFPN.FUSION.EPS)
            m_fused  = self.conv[f"m{level_high}"](m_fused)

            m[level_high] = m_fused

        # ___/ Stage Q \___
        # Fuse features from high resolution to low resolution (pathway Q)...
        q = {}
        for idx, level_high in enumerate(range(min_level, max_level)):
            level_low = level_high + 1
            q_high = m[level_high] if idx == 0              else q[level_high]
            m_low  = m[level_low ] if level_low < max_level else p[level_low ]
            p_low  = p[level_low ]

            w1, w2, w3 = self.w_q[idx]
            q_high_up = F.interpolate(q_high,
                                      scale_factor  = bifpn_internal_config.MODEL.BIFPN.DOWN_SCALE_FACTOR,
                                      mode          = 'bilinear',
                                      align_corners = False)
            q_fused  = w1 * p_low + w2 * m_low + w3 * q_high_up
            q_fused /= (w1 + w2 + w3 + bifpn_internal_config.MODEL.BIFPN.FUSION.EPS)
            q_fused  = self.conv[f"q{level_low}"](q_fused)

            if idx == 0: q[level_high] = q_high
            q[level_low] = q_fused

        return [ q[level] for level in range(min_level, min_level + num_levels) ]




class BiFPN(nn.Module):
    """
    This class provides a series of BiFPN blocks.
    """

    def __init__(self, num_blocks = 1, num_features = 64, num_levels = 5):
        super().__init__()

        self.blocks = nn.Sequential(*[
            BiFPNBlock(num_features = num_features,
                       num_levels = num_levels)
            for block_idx in range(num_blocks)
        ])


    def forward(self, x):
        return self.blocks(x)
