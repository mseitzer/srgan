import functools

import torch.nn.init

DEFAULT_INITS = {
    'conv_weight': ('he_normal', 0.0),
    'conv_bias': ('constant', 0.0),
    'conv_transposed_weight': ('he_normal', 0.0),
    'conv_transposed_bias': ('constant', 0.0),
    'batchnorm_weight': ('constant', 1.0),
    'batchnorm_bias': ('constant', 0.0),
    'linear_weight': ('xavier_normal', 'linear'),
    'linear_bias': ('constant', 0.0,)
}


def _get_init_fn(init):
  init_name = init[0] if isinstance(init, (tuple, list)) else init

  if init_name == 'torch_default':
    return lambda _: None  # Do nothing

  elif init_name == 'zero':
    return functools.partial(torch.nn.init.constant, val=0.0)

  elif init_name == 'constant':
    return functools.partial(torch.nn.init.constant, val=init[1])

  elif init_name == 'normal':
    assert len(init) == 3, 'Need mean and std for normal init'
    return functools.partial(torch.nn.init.normal, mean=init[1], std=init[2])

  elif init_name == 'uniform':
    assert len(init) == 3, 'Need lower and upper value for uniform init'
    return functools.partial(torch.nn.init.uniform, a=init[1], b=init[2])

  elif init_name.startswith('xavier'):
    assert isinstance(init, (tuple, list)), 'Need gain value for Xavier init'
    gain = init[1]
    if isinstance(init[1], str):
      gain = torch.nn.init.calculate_gain(init[1])

    if init_name == 'xavier_normal':
      return functools.partial(torch.nn.init.xavier_normal, gain=gain)
    else:
      return functools.partial(torch.nn.init.xavier_uniform, gain=gain)

  elif init_name.startswith('he'):
    # a: negative slope of the used rectifier unit
    a = init[1] if isinstance(init, (tuple, list)) else 0.0
    if init_name == 'he_normal':
      return functools.partial(torch.nn.init.kaiming_normal, a=a)
    else:
      return functools.partial(torch.nn.init.kaiming_uniform, a=a)

  elif init_name == 'orthogonal':
    gain = init[1] if isinstance(init, (tuple, list)) else 1.0
    if isinstance(gain, str):
      gain = torch.nn.init.calculate_gain(gain)
    return functools.partial(torch.nn.init.orthogonal, gain=gain)

  else:
    assert False, 'Unknown weight init {}'.format(init_name)


def _weight_init(init_params, m):
  def get_inits(m, init_params, weight_key, bias_key):
    weight_init, bias_init = None, None
    if weight_key in init_params and m.weight is not None:
      weight_init = init_params[weight_key]
    if bias_key in init_params and m.bias is not None:
      bias_init = init_params[bias_key]

    return weight_init, bias_init

  weight_init, bias_init = None, None
  classname = m.__class__.__name__

  if m in init_params:
    # Allow specialized init of individual modules
    # This is kind of a hack, because we can't specify this through the
    # configuration. But our layers have no names, so we can't address
    # individual ones.
    weight_init, bias_init = get_inits(m, init_params[m],
                                       'weight', 'bias')
  elif classname.find('Conv2d') != -1:
    weight_init, bias_init = get_inits(m, init_params,
                                       'conv_weight', 'conv_bias')
  elif classname.find('ConvTranspose2d') != -1:
    weight_init, bias_init = get_inits(m, init_params,
                                       'conv_transposed_weight',
                                       'conv_transposed_bias')
  elif classname.find('Linear') != -1:
    weight_init, bias_init = get_inits(m, init_params,
                                       'linear_weight', 'linear_bias')
  elif classname.find('BatchNorm2d') != -1:
    weight_init, bias_init = get_inits(m, init_params,
                                       'batchnorm_weight', 'batchnorm_bias')

  if weight_init is not None:
    _get_init_fn(weight_init)(m.weight.data)
  if bias_init is not None:
    _get_init_fn(bias_init)(m.bias.data)


def initialize_weights(model, conf_weight_init):
  init_params = DEFAULT_INITS.copy()
  init_params.update(model.weight_init_params())
  init_params.update(conf_weight_init)

  model.apply(functools.partial(_weight_init, init_params))
