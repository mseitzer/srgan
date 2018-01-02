import math

import torch.nn as nn


def get_activation_fn(name, leakiness=None, num_channels=None):
  if name == 'relu':
    return nn.ReLU()
  elif name == 'prelu':
    assert num_channels is not None
    assert leakiness is not None
    return nn.PReLU(num_parameters=num_channels, init=leakiness)
  elif name == 'lrelu':
    assert leakiness is not None
    return nn.LeakyReLU(negative_slope=leakiness)
  elif name == 'softmax':
    return nn.Softmax()
  elif name == 'tanh':
    return nn.Tanh()
  else:
    assert False, 'Unknown activation function {}'.format(name)


def get_normalization_layer(name, num_features):
  if name == 'batch':
    return nn.BatchNorm2d(num_features, affine=True)
  elif name == 'instance':
    return nn.InstanceNorm2d(num_features, affine=False)
  elif name == 'instance-affine':
    return nn.InstanceNorm2d(num_features, affine=True)
  else:
    raise ValueError('Unknown normalization layer {}'.format(name))


def need_bias(use_norm_layers, norm_layer):
  if not use_norm_layers or \
     use_norm_layers == 'not-first' or \
     norm_layer == 'instance':
    return True
  elif norm_layer == 'batch' or norm_layer == 'instance-affine':
    return False
  else:
    return False


def get_padding_layer(total_padding, mode='zero'):
  padding_layers = {
      'zero': nn.ZeroPad2d,
      'reflection': nn.ReflectionPad2d,
      'replication': nn.ReplicationPad2d
  }
  assert mode in padding_layers

  padding_side = total_padding // 2
  if total_padding % 2 == 0:
    padding = padding_side
  else:
    padding = (padding_side, padding_side + 1, padding_side, padding_side + 1)

  return padding_layers[mode](padding)


def get_same_padding_layer(kernel_size, stride, mode='zero'):
  """Constructs padding layer for SAME padding

  Calculates padding to insert such that the spatial dimensions stay the same
  after a 2d convolution.
  WARNING: Only works for even sized input sizes and stride one or two.
  """
  assert stride == 1 or stride == 2, 'Formula only works for stride 1 or 2'
  total_padding = int(math.ceil(kernel_size / stride)) - 1
  return get_padding_layer(total_padding, mode)
