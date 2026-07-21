from functools import partial
import torch.nn as nn

from model_oldRepo import mamba_extractor  # already the `feature_extractor` class (see model_oldRepo/__init__.py)
from model_oldRepo.accel_flow import AccelFlow
# NOTE: model_oldRepo/__init__.py aliases `mamba_estimation` to
# flow_estimation.MultiScaleFlow (the bidirectional-flow model -- correct,
# not stale) -- but that class has no accel_head, so going through that
# alias silently disables the learned-acceleration supervision described in
# the requirements doc (train.py's `hasattr(model.net, 'accel_head')` gate
# would never trigger). Import AccelFlow directly instead if you want the
# acceleration term active.

LOG = 'VFIMamba'
LOCAL = 2

'''==========Model config=========='''
def init_model_config(F=32, W=7, depth=[2, 2, 2, 4, 4], M=False, accel_scale=1.0, accel_hidden=48):
    '''This function should not be modified'''
    return {
        'embed_dims':[(2**i)*F for i in range(len(depth))],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads':[8*(2**i)*F//32 for i in range(len(depth)-3)],
        'mlp_ratios':[4 for i in range(len(depth)-3)],
        'qkv_bias':True,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6),
        'depths':depth,
        'window_sizes':[W for i in range(len(depth)-3)],
        'conv_stages':3
    }, {
        'embed_dims':[(2**i)*F for i in range(len(depth))],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'depths':depth,
        'num_heads':[8*(2**i)*F//32 for i in range(len(depth)-3)],
        'window_sizes':[W, W],
        'scales':[4*(2**i) for i in range(len(depth)-2)],
        'hidden_dims':[4*F for i in range(len(depth)-3)],
        'c':F,
        'M':M,
        'local_hidden_dims':4*F,
        'local_num':2,
        'accel_scale': accel_scale,   # AccelHead's tanh output scale -- new kwarg AccelFlow.__init__ expects
        'accel_hidden': accel_hidden, # AccelHead's hidden channel width (defaults to 48 if omitted)
    }


MODEL_CONFIG = {
    'LOGNAME': LOG,
    'MODEL_TYPE': (mamba_extractor, AccelFlow),  # AccelFlow, not the mamba_estimation alias -- see import comment above
    'MODEL_ARCH': init_model_config(
        F = 32,
        depth = [2, 2, 2, 3, 3],
        M = False,
        accel_scale = 1.0,
        accel_hidden = 48,
    )
}