import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.utils import (get_activation_fn, get_same_padding_layer,
                          get_normalization_layer, need_bias)
from models.weight_inits import initialize_weights


REQUIRED_PARAMS = [
    'num_inputs', 'num_filters_per_layer', 'strides'
]

OPTIONAL_PARAMS = [
    'kernel_sizes', 'fc_layers', 'spatial_shape', 'act_fn',
    'relu_leakiness', 'use_norm_layers', 'norm_layer', 'padding'
]


def construct_model(conf, model_name):
  if model_name == 'CNNDiscriminator':
    params = conf.to_param_dict(REQUIRED_PARAMS, OPTIONAL_PARAMS)
    model = CNNDiscriminator(**params)
    initialize_weights(model, conf.get_attr('weight_init', default={}))
  else:
    raise ValueError('Unknown discriminator {}'.format(model_name))
  return model


class CNNDiscriminator(nn.Module):
  """CNN based discriminator network"""
  DEFAULT_RELU_LEAKINESS = 0.2

  def __init__(self, num_inputs, num_filters_per_layer, strides,
               kernel_sizes=None, fc_layers=[], spatial_shape=None,
               act_fn='lrelu', relu_leakiness=DEFAULT_RELU_LEAKINESS,
               use_norm_layers=True, norm_layer='batch', padding='zero'):
    """Construct model

    Parameters
    ----------
    num_inputs: int
      Number of input channels
    num_filters_per_layer : list or tuple
      Number of filters the discriminator uses in each layer
    kernel_sizes : list
      Shape of filters in each layer. Defaults to 3
    strides : list
      Strides of filters in each layer.
    fc_layers : list
      Number of channels of fully connected layers after convolutional layers.
      If no fully connected layers are selected, the convolutional features
      maps will be reduced to one dimension with a 1x1 convolution, and the
      output is a probability map (corresponds to a PatchGAN)
    spatial_shape : tuple
      Spatial shape of input in the form of (height, width). Required if
      using fully connected layers
    act_fn : string
      Activation function to use. Either `relu`, `prelu`, or `lrelu` (default)
    relu_leakiness : float
      If using lrelu, leakiness of the relus, if using prelu, initial value
      for prelu parameters
    use_norm_layers : bool or string
      If true, use normalization layers. If `not-first`, skip the normalization
      after the first convolutional layer
    norm_layer : string
      Normalization layer to use. `batch` for batch normalization or `instance`
      for instance normalization
    padding : string
      Type of padding to use. Either `zero`, `reflection`, or `replication`
    """
    super(CNNDiscriminator, self).__init__()
    if len(fc_layers) > 0:
      assert spatial_shape is not None, \
          'Need input spatial shape if using fully connected layers'

    if kernel_sizes is None:
      kernel_sizes = 3
    if isinstance(kernel_sizes, int):
      kernel_sizes = [kernel_sizes] * len(num_filters_per_layer)

    in_channels = num_inputs

    model = []
    for num_filters, kernel_size, stride in zip(num_filters_per_layer,
                                                kernel_sizes,
                                                strides):
      model += (get_same_padding_layer(kernel_size=kernel_size, stride=stride,
                                       mode=padding),
                nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size,
                          stride=stride,
                          bias=need_bias(use_norm_layers, norm_layer)))

      if use_norm_layers != 'not-first' and use_norm_layers:
        model.append(get_normalization_layer(norm_layer, num_filters))
      elif use_norm_layers == 'not-first':
        use_norm_layers = True
      model.append(get_activation_fn(act_fn, relu_leakiness, num_filters))
      in_channels = num_filters

    self.convs = nn.Sequential(*model)

    if len(fc_layers) > 0:
      input_dims = self._infer_shape(self.convs, num_inputs, spatial_shape)

      model = []
      for num_features in fc_layers[:-1]:
        model += (nn.Linear(input_dims, num_features),
                  get_activation_fn(act_fn, relu_leakiness, num_features))
        input_dims = num_features

      model.append(nn.Linear(input_dims, fc_layers[-1]))

      self.fcs = nn.Sequential(*model)
      self.final_conv = None
    else:
      self.fcs = None
      self.final_conv = nn.Conv2d(in_channels, out_channels=1,
                                  kernel_size=1, stride=1, bias=True)

  @staticmethod
  def _infer_shape(model, num_inputs, spatial_shape):
    """Infer shape by doing a forward pass"""
    inp = Variable(torch.ones(1, num_inputs,
                              spatial_shape[0], spatial_shape[1]),
                   volatile=True)
    outp = model(inp)
    return outp.view(1, -1).shape[1]

  def weight_init_params(self):
    return {
        'conv_weight': ('normal', 0.0, 0.02),
        'linear_weight': ('normal', 0.0, 0.02),
        'batchnorm_weight': ('normal', 1.0, 0.02)
    }

  def forward(self, x):
    x = self.convs(x)

    if self.fcs is not None:
      x = x.view(x.shape[0], -1)
      x = self.fcs(x)
    else:
      x = self.final_conv(x)

    prob = F.sigmoid(x)

    return {
        'prob': prob
    }
