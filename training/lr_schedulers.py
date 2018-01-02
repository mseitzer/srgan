import torch.optim.lr_scheduler as lr_scheduler


def _get_polynomial_decay(lr, end_lr, decay_epochs, from_epoch=0, power=1.0):
  # Note: epochs are zero indexed by pytorch
  def lr_lambda(epoch):
    if epoch < from_epoch:
      return 1.0
    epoch = min(epoch, end_epoch)
    new_lr = ((lr - end_lr) * (1. - epoch / end_epoch) ** power + end_lr)
    return new_lr / lr  # LambdaLR expects returning a factor

  end_epoch = float(from_epoch + decay_epochs)
  return lr_lambda


def is_pre_epoch_scheduler(scheduler):
  return (scheduler is not None and
          not isinstance(scheduler, lr_scheduler.ReduceLROnPlateau))


def is_post_epoch_scheduler(scheduler):
  return isinstance(scheduler, lr_scheduler.ReduceLROnPlateau)


def get_lr_scheduler(optimizer_conf, scheduler_name, optimizer, initial_epoch=-1):
  if scheduler_name == 'multistep':
    return lr_scheduler.MultiStepLR(optimizer,
                                    optimizer_conf.decay_steps,
                                    optimizer_conf.decay_factor,
                                    initial_epoch)
  elif scheduler_name == 'linear' or scheduler_name == 'polynomial':
    power = 1.0 if scheduler_name == 'linear' else optimizer_conf.decay_power
    lr_lambda = _get_polynomial_decay(optimizer_conf.learning_rate,
                                      optimizer_conf.end_learning_rate,
                                      optimizer_conf.decay_steps,
                                      optimizer_conf.get_attr('start_decay',
                                                              default=0),
                                      power)
    return lr_scheduler.LambdaLR(optimizer, lr_lambda, initial_epoch)
  else:
    raise ValueError('Unknown learning rate scheduler {}'.format(scheduler_name))


