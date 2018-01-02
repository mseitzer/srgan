"""Model performing just bilinear upsampling"""
import torch.nn as nn

from utils.tensor_transforms import scale_to_range

CONFIG_KEYS = [
    'upscale_factor'
]

REQUIRED_PARAMS = [
    'upscale_factor'
]


def construct_model(conf, model_name):
  params = conf.to_param_dict(CONFIG_KEYS, REQUIRED_PARAMS)
  return BilinearUpsample(**params)


class BilinearUpsample(nn.Module):
  def __init__(self, upscale_factor):
    super(BilinearUpsample, self).__init__()
    self.upsample = nn.Upsample(scale_factor=int(upscale_factor),
                                mode='bilinear')

  def forward(self, x):
    x = self.upsample(x)
    x = scale_to_range(x, (-1., 1.))
    return x
