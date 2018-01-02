import torch.optim as optim


def get_optimizer(conf, optimizer_name, variables):
  if optimizer_name == 'Adam':
    beta1 = conf.get_attr('beta1', default=0.9)
    beta2 = conf.get_attr('beta2', default=0.999)
    return optim.Adam(variables, conf.learning_rate, betas=(beta1, beta2))
  else:
    raise ValueError('Unknown optimizer {}'.format(optimizer_name))
