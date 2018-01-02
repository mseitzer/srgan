import torch.nn as nn

from models.utils import (get_activation_fn, get_same_padding_layer,
                          get_normalization_layer, need_bias)
from models.weight_inits import initialize_weights

REQUIRED_PARAMS = [
    'num_inputs', 'num_outputs', 'upscale_factor'
]

OPTIONAL_PARAMS = [
    'num_filters', 'num_res_blocks',
    'output_activation', 'act_fn', 'relu_leakiness',
    'use_norm_layers', 'norm_layer', 'padding'
]


def construct_model(conf, model_name):
  params = conf.to_param_dict(REQUIRED_PARAMS, OPTIONAL_PARAMS)
  model = SRResNet(**params)
  initialize_weights(model, conf.get_attr('weight_init', default={}))
  return model


class ResBlock(nn.Module):
  def __init__(self, in_channels, num_filters, kernel_size, use_norm_layers,
               norm_layer, act_fn, relu_leakiness=None, padding='zero'):
    """Builds a residual block

    The implementation follows the SRGAN paper, which uses a residual block
    variant which uses no activation after the addition. This design is based
    on Sam Gross & Michael Wilber: "Training and investigating Residual Nets"
    (see http://torch.ch/blog/2016/02/04/resnets.html)

    Parameters
    ----------
    in_channels : int
      Number of input channels
    num_filters : int
      Number of convolutional filters to use
    kernel_size : int
      Size of convolution kernel
    use_norm_layers : bool
      If true, uses normalization layers after the convolution layers
    norm_layer : string
      Normalization layer to use. `batch` for batch normalization or `instance`
      for instance normalization
    act_fn : string
      Activation function to use. Either `relu`, `prelu` (default), or `lrelu`
    relu_leakiness : float
      If using lrelu, leakiness of the relus, if using prelu, initial value
      for prelu parameters
    padding : string
      Type of padding to use. Either `zero`, `reflection`, or `replication`
    """
    super(ResBlock, self).__init__()
    use_bias = need_bias(use_norm_layers, norm_layer)
    modules = [get_same_padding_layer(kernel_size, stride=1, mode=padding),
               nn.Conv2d(in_channels, num_filters, kernel_size=kernel_size,
                         stride=1, bias=use_bias)]
    if use_norm_layers:
      modules.append(get_normalization_layer(norm_layer, num_filters))

    modules += [get_activation_fn(act_fn, relu_leakiness, num_filters),
                get_same_padding_layer(kernel_size, stride=1, mode=padding),
                nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size,
                          stride=1, bias=use_bias)]
    if use_norm_layers:
      modules.append(get_normalization_layer(norm_layer, num_filters))

    self.block = nn.Sequential(*modules)

  def forward(self, x):
    return self.block(x) + x


class SRResNet(nn.Module):
  DEFAULT_RELU_LEAKINESS = 0.1

  def __init__(self, num_inputs, num_outputs, upscale_factor,
               num_filters=64, num_res_blocks=16,
               output_activation='tanh', act_fn='prelu',
               relu_leakiness=DEFAULT_RELU_LEAKINESS,
               use_norm_layers='not-first', norm_layer='batch',
               padding='zero'):
    """Builds a SRResNet (Ledig et al, https://arxiv.org/abs/1609.04802)

    Parameters
    ----------
    num_inputs : int
      Number of input channels
    num_outputs : int
      Number of output channels
    upscale_factor : int
      Factor by which the network upscales. Must be divisible by 2 or 3
    num_filters : int
      Number of convolutional filters to use
    num_res_blocks : int
      Number of residual blocks
    output_activation : string
      Either `softmax` or `tanh`. Activation function to use on the logits
    act_fn : string
      Activation function to use. Either `relu`, `prelu` (default), or `lrelu`
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
    super(SRResNet, self).__init__()
    upscale_factor = int(upscale_factor)
    assert (upscale_factor == 1 or
            upscale_factor % 2 == 0 or
            upscale_factor % 3 == 0)
    in_channels = num_inputs

    initial_conv = [get_same_padding_layer(kernel_size=9, stride=1,
                                           mode=padding),
                    nn.Conv2d(in_channels, num_filters, kernel_size=9,
                              stride=1,
                              bias=need_bias(use_norm_layers, norm_layer)),
                    get_activation_fn(act_fn, relu_leakiness, num_filters)]
    in_channels = num_filters

    if use_norm_layers != 'not-first' and use_norm_layers:
      initial_conv.append(get_normalization_layer(norm_layer, in_channels))
    elif use_norm_layers == 'not-first':
      use_norm_layers = True

    res_blocks = []
    for idx in range(num_res_blocks):
      res_blocks += [ResBlock(in_channels, num_filters, kernel_size=3,
                              use_norm_layers=use_norm_layers,
                              norm_layer=norm_layer, act_fn=act_fn,
                              relu_leakiness=relu_leakiness)]

    second_conv = [get_same_padding_layer(kernel_size=3, stride=1,
                                          mode=padding),
                   nn.Conv2d(in_channels, num_filters, kernel_size=3,
                             stride=1, bias=need_bias(use_norm_layers,
                                                      norm_layer))]
    in_channels = num_filters
    if use_norm_layers:
      second_conv.append(get_normalization_layer(norm_layer, in_channels))

    upsample = []
    if upscale_factor > 1:
      scale = 2 if upscale_factor % 2 == 0 else 3
      for idx in range(upscale_factor // scale):
        upsample += [get_same_padding_layer(kernel_size=3, stride=1,
                                            mode=padding),
                     nn.Conv2d(in_channels, scale * scale * 256,
                               kernel_size=3, stride=1, bias=True),
                     nn.PixelShuffle(upscale_factor=scale),
                     get_activation_fn(act_fn, relu_leakiness, 256)]
        in_channels = 256

    final_conv = [get_same_padding_layer(kernel_size=9, stride=1,
                                         mode=padding),
                  nn.Conv2d(in_channels, num_outputs, kernel_size=9,
                            stride=1, bias=True)]
    if output_activation != 'none':
      final_conv.append(get_activation_fn(output_activation))

    self.initial_conv = nn.Sequential(*initial_conv)
    self.body = nn.Sequential(*(res_blocks + second_conv))
    self.upsample = nn.Sequential(*upsample)
    self.final_conv = nn.Sequential(*final_conv)
    self.output_activation = output_activation

  def weight_init_params(self):
    init_params = {
        'conv_weight': ('orthogonal', 'relu')
    }

    # Special case for last convolution
    conv = self.final_conv[1]
    assert isinstance(conv, nn.Conv2d)
    if self.output_activation == 'none':
      init_params[conv] = {'weight': 'orthogonal'}
    else:
      init_params[conv] = {'weight': ('orthogonal', self.output_activation)}

    return init_params

  def forward(self, x):
    initial = self.initial_conv(x)
    x = self.body(initial)
    x = self.upsample(x + initial)
    x = self.final_conv(x)
    return x

