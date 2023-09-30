
def prepare_Slice2RotMat_BIFPN_inputs(config, use_bifpn=False):
    if use_bifpn:
        out_config = {
            'size': config.MODEL.BACKBONE.RES_TYPE, 
            'pretrained': config.MODEL.BACKBONE.PRETRAIN,
            'num_features': config.MODEL.BIFPN.NUM_FEATURES,
            'num_blocks': config.MODEL.BIFPN.NUM_BLOCKS,
            'output_channels': config.MODEL.BACKBONE.OUTPUT_CHANNELS,
            'num_levels': config.MODEL.BIFPN.NUM_LEVELS,
            'regressor_in_features': config.MODEL.REGRESSOR_HEAD.IN_FEATURES,
            'regressor_out_features': config.MODEL.REGRESSOR_HEAD.OUT_FEATURES,
            'scale': config.MODEL.RESNET2ROTMAT.SCALE
        }
    else:
        out_config = {
            'size': config.MODEL.BACKBONE.RES_TYPE, 
            'pretrained': config.MODEL.BACKBONE.PRETRAIN
        }
    return out_config