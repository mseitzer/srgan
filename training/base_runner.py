import os
from itertools import chain

import numpy as np
import torch
from torch.autograd import Variable

import utils
from metrics import accumulate_metric
from utils.checkpoints import load_model_state_dict


class BaseRunner(object):
  """Abstract base class for running training and validation"""
  def __init__(self, cuda=''):
    self.cuda = cuda

    self.data_iter = None

  def _get_loss_weights(self, weights_by_criterion, *args):
    weights = [weights_by_criterion.get(name, 1.0)
               for criteria in args for name in criteria]
    if len(weights) == 0:
      return None
    tensor = torch.from_numpy(np.array(weights, dtype=np.float32))
    if self.cuda != '':
      tensor = utils.cudaify(tensor)
    return Variable(tensor, requires_grad=False)

  def _request_data(self, loader, volatile=False):
    try:
      batch = next(self.data_iter)
    except StopIteration:
      self.data_iter = None
      return None

    batch = [Variable(tensor, volatile=volatile) for tensor in batch]

    if self.cuda != '':
      batch = utils.cudaify(batch)

    return batch

  def train_epoch(self, loader, epoch, summary_writer=None,
                  steps_per_train_summary=1, verbose=False):
    num_batches_per_epoch = len(loader)
    epoch_loss_metrics = {}
    epoch_metrics = {}
    self._set_train()

    self.data_iter = iter(loader)

    current_batch = 0
    while current_batch < num_batches_per_epoch:
      num_batches, loss_metrics, data = self._train_step(loader)
      if num_batches == 0:
        break

      current_batch += num_batches

      metrics = self._compute_train_metrics(data)

      for name, loss_metric in loss_metrics.items():
        accumulate_metric(epoch_loss_metrics, name, loss_metric)
      for name, metric in metrics.items():
        accumulate_metric(epoch_metrics, name, metric)

      global_step = num_batches_per_epoch * (epoch - 1) + current_batch
      if global_step % steps_per_train_summary == 0:
        s = '===> Epoch[{}]({}/{}): '.format(epoch, current_batch,
                                             num_batches_per_epoch)
        s += ', '.join(('{}: {}'.format(name, loss_metric)
                        for name, loss_metric in loss_metrics.items()))
        s += '\n'
        if verbose:
          s += '\n'.join(('     {}: {}'.format(name, metric)
                         for name, metric in metrics.items()))
        print(s)

        if summary_writer is not None:
          for name, metric in chain(loss_metrics.items(), metrics.items()):
            summary_writer.add_scalar('train/{}'.format(name), metric.value,
                                      global_step)

    value_by_loss = {loss: loss_value.average()
                     for loss, loss_value in epoch_loss_metrics.items()}
    value_by_metric = {metric: metric_value.average()
                       for metric, metric_value in epoch_metrics.items()}
    return value_by_loss, value_by_metric

  def validate(self, loader, num_batches_to_return=0):
    epoch_data = []
    epoch_loss_metrics = {}
    epoch_metrics = {}
    self._set_test()

    self.data_iter = iter(loader)

    for current_batch in range(len(loader)):
      loss_metrics, data = self._val_step(loader)
      if data is None:
        break

      if len(epoch_data) < num_batches_to_return:
        epoch_data.append(utils.cpuify(data))

      metrics = self._compute_test_metrics(data)

      for name, loss_metric in loss_metrics.items():
        accumulate_metric(epoch_loss_metrics, name, loss_metric)
      for name, metric in metrics.items():
        accumulate_metric(epoch_metrics, name, metric)

    value_by_loss = {name: loss_metric.average()
                     for name, loss_metric in epoch_loss_metrics.items()}
    value_by_metric = {name: metric.average()
                       for name, metric in epoch_metrics.items()}

    return epoch_data, value_by_loss, value_by_metric

  def infer(self, loader):
    epoch_data = []
    self._set_test()

    self.data_iter = iter(loader)

    for current_batch in range(len(loader)):
      _, data = self._val_step(loader, compute_metrics=False)
      if data is None:
        break

      epoch_data.append(utils.cpuify(data))

    return epoch_data

  @staticmethod
  def initialize_pretrained_model(model_conf, model, cuda, conf_path):
    path, model_key = model_conf.pretrained_weights
    if not os.path.isabs(path):
      path = os.path.join(os.path.dirname(conf_path), path)
    state_dict = load_model_state_dict(path, model_key, cuda)
    model.load_state_dict(state_dict)
    print(('Loaded pretrained weights from '
           'checkpoint {}, key {}').format(path, model_key))

  def get_named_outputs(self, data):
    """Translate data output of validate method to dictionary"""
    raise NotImplementedError('Subclasses must override get_named_outputs')

  def state_dict(self):
    """Get state of this runner"""
    raise NotImplementedError('Subclasses must override state_dict')

  def load_state_dict(self, state_dict):
    raise NotImplementedError('Subclasses must override load_state_dict')

  def epoch_finished(self, epoch):
    raise NotImplementedError('Subclasses must override epoch_finished')

  def _train_step(self, loader):
    raise NotImplementedError('Subclasses must override _train_steps')

  def _val_step(self, loader, compute_metrics):
    raise NotImplementedError('Subclasses must override _val_step')

  def _compute_train_metrics(self, data):
    raise NotImplementedError(('Subclasses must override '
                               '_compute_train_metrics'))

  def _compute_test_metrics(self, data):
    raise NotImplementedError(('Subclasses must override '
                               '_compute_test_metrics'))

  def _set_train(self):
    raise NotImplementedError('Subclasses must override _set_train')

  def _set_test(self):
    raise NotImplementedError('Subclasses must override _set_test')
