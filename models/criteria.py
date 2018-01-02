import torch.nn as nn


def _get_adv_criterion(conf, loss_name, cuda, loss_type):
  from models.adversarial_loss import AdversarialLoss
  return AdversarialLoss(loss_name, loss_type, cuda)


def _get_vgg_criterion(conf, loss_name, cuda):
  from models.vgg_loss import VGGLoss
  return VGGLoss(loss_name, cuda,
                 conf.get_attr('vgg_loss_blocks', default=-1),
                 conf.get_attr('vgg_loss_criterion', default='MSE'),
                 conf.get_attr('vgg_loss_weights'))


_CRITERIA = {
    'MSE': nn.MSELoss,
    'L1': nn.L1Loss,
    'SmoothL1Loss': nn.SmoothL1Loss,
    'gan': _get_adv_criterion,
    'VGG19': _get_vgg_criterion
}


def get_criterion(conf, loss_name, *args):
  assert loss_name in _CRITERIA, 'Unknown loss {}'.format(loss_name)
  criterion = _CRITERIA[loss_name]
  if isinstance(criterion, type):
    # Class: probably directly pytorch criterion
    return criterion()
  else:
    # Function: pass additional information
    return criterion(conf, loss_name, *args)

