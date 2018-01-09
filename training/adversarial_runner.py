from collections import OrderedDict

import torch

import utils
from data.transform_wrappers import get_output_transform
from metrics import get_metric_fn, get_loss_metric, accumulate_metric
from models import construct_model
from models.criteria import get_criterion
from training.lr_schedulers import (get_lr_scheduler,
                                    is_pre_epoch_scheduler,
                                    is_post_epoch_scheduler)
from training.optimizers import get_optimizer
from training.base_runner import BaseRunner
from utils.config import Configuration

DEFAULT_INPUT_METHOD = 'simple'


def build_runner(conf, cuda, mode, resume=False):
  gen_model_conf = Configuration.from_dict(conf.generator_model)
  gen_model = construct_model(gen_model_conf, gen_model_conf.name)

  val_metric_transform = get_output_transform(conf, conf.application, 'test')
  val_metric_fns = {name: get_metric_fn(name)
                    for name in conf.get_attr('validation_metrics',
                                              default=[])}
  output_transform = get_output_transform(conf, conf.application, 'output')

  if mode == 'train':
    disc_model_conf = Configuration.from_dict(conf.discriminator_model)
    disc_model = construct_model(disc_model_conf, disc_model_conf.name)

    gen_adv_criteria = {loss_name: get_criterion(conf, loss_name, cuda, 'gen')
                        for loss_name in conf.generator_adversarial_losses}
    gen_criteria = {loss_name: get_criterion(conf, loss_name, cuda)
                    for loss_name in conf.generator_losses}
    disc_adv_criteria = {loss_name: get_criterion(conf, loss_name, cuda,
                                                  'disc')
                         for loss_name in conf.discriminator_losses}

    if cuda != '':
      utils.cudaify([gen_model, disc_model] +
                    list(gen_adv_criteria.values()) +
                    list(gen_criteria.values()) +
                    list(disc_adv_criteria.values()))

    # Important: construct optimizers after moving model to GPU!
    gen_opt_conf = Configuration.from_dict(conf.generator_optimizer)
    gen_optimizer = get_optimizer(gen_opt_conf, gen_opt_conf.name,
                                  gen_model.parameters())
    gen_lr_scheduler = None
    if gen_opt_conf.has_attr('lr_scheduler'):
      gen_lr_scheduler = get_lr_scheduler(gen_opt_conf,
                                          gen_opt_conf.lr_scheduler,
                                          gen_optimizer)

    disc_opt_conf = Configuration.from_dict(conf.discriminator_optimizer)
    disc_optimizer = get_optimizer(disc_opt_conf, disc_opt_conf.name,
                                   disc_model.parameters())
    disc_lr_scheduler = None
    if disc_opt_conf.has_attr('lr_scheduler'):
      disc_lr_scheduler = get_lr_scheduler(disc_opt_conf,
                                           disc_opt_conf.lr_scheduler,
                                           disc_optimizer)

    train_disc_metrics = conf.get_attr('train_discriminator_metrics',
                                       default=[])
    train_disc_metric_fns = {name: get_metric_fn(name)
                             for name in train_disc_metrics}

    train_gen_metric_transform = get_output_transform(conf, conf.application,
                                                      'train')
    train_gen_metrics = conf.get_attr('train_generator_metrics', default=[])
    train_gen_metric_fns = {name: get_metric_fn(name)
                            for name in train_gen_metrics}

    input_method = disc_model_conf.get_attr('input_method',
                                            default=DEFAULT_INPUT_METHOD)

    runner = AdversarialRunner(gen_model, disc_model,
                               gen_optimizer, disc_optimizer,
                               gen_lr_scheduler, disc_lr_scheduler,
                               gen_adv_criteria, gen_criteria,
                               disc_adv_criteria,
                               conf.get_attr('generator_loss_weights', {}),
                               conf.get_attr('discriminator_loss_weights', {}),
                               cuda,
                               train_gen_metric_fns,
                               train_gen_metric_transform,
                               train_disc_metric_fns,
                               val_metric_fns,
                               val_metric_transform,
                               output_transform,
                               input_method)
    if gen_model_conf.has_attr('pretrained_weights') and not resume:
      runner.initialize_pretrained_model(gen_model_conf, runner.gen,
                                         cuda, conf.file)

    if disc_model_conf.has_attr('pretrained_weights') and not resume:
      runner.initialize_pretrained_model(disc_model_conf, runner.disc,
                                         cuda, conf.file)
  else:
    if cuda != '':
      utils.cudaify(gen_model)
    runner = AdversarialRunner(gen_model,
                               cuda=cuda,
                               val_metric_fns=val_metric_fns,
                               val_metric_transform=val_metric_transform,
                               output_transform=output_transform)

  return runner


def _get_disc_input_fn(method):
  def simple(out_gen, inp, detach=False):
    return out_gen.detach() if detach else out_gen

  def concat(out_gen, inp, detach=False):
    if detach:
      out_gen = out_gen.detach()
    return torch.cat((out_gen.detach(), inp), dim=1)

  if method == 'simple':
    return simple
  elif method == 'concat':
    return concat
  else:
    raise ValueError('Unknown discriminator input method {}'.format(method))


class AdversarialRunner(BaseRunner):
  """A runner for an adversarial model with generator and discriminator"""
  def __init__(self, gen_model, disc_model=None,
               gen_optimizer=None, disc_optimizer=None,
               gen_lr_scheduler=None, disc_lr_scheduler=None,
               gen_adv_criteria={}, gen_criteria={}, disc_adv_criteria={},
               gen_loss_weights={}, disc_loss_weights={}, cuda='',
               train_gen_metric_fns={}, train_gen_metric_transform=None,
               train_disc_metric_fns={},
               val_metric_fns={}, val_metric_transform=None,
               output_transform=None,
               disc_input_method=DEFAULT_INPUT_METHOD):
    super(AdversarialRunner, self).__init__(cuda)
    self.gen = gen_model
    self.disc = disc_model
    self.gen_optimizer = gen_optimizer
    self.disc_optimizer = disc_optimizer
    self.gen_lr_scheduler = gen_lr_scheduler
    self.disc_lr_scheduler = disc_lr_scheduler

    self.train_gen_metric_fns = train_gen_metric_fns
    self.train_gen_metric_transform = train_gen_metric_transform
    self.train_disc_metric_fns = train_disc_metric_fns
    self.val_metric_fns = val_metric_fns
    self.val_metric_transform = val_metric_transform
    self.output_transform = output_transform

    self.disc_input_fn = _get_disc_input_fn(disc_input_method)

    self.gen_adv_criteria = OrderedDict(gen_adv_criteria)
    self.gen_criteria = OrderedDict(gen_criteria)
    self.disc_adv_criteria = OrderedDict(disc_adv_criteria)

    self.gen_loss_weights = self._get_loss_weights(gen_loss_weights,
                                                   gen_adv_criteria,
                                                   gen_criteria)
    self.disc_loss_weights = self._get_loss_weights(disc_loss_weights,
                                                    disc_adv_criteria)

  def get_named_outputs(self, data):
    prediction, target = data[1], data[2]
    if self.output_transform is not None:
      prediction, target = self.output_transform(prediction, target)

    return {
        'input': data[0],
        'prediction': prediction,
        'target': target,
        'disc_fake': data[3]
    }

  def state_dict(self):
    return {
        'generator': self.gen.state_dict(),
        'discriminator': self.disc.state_dict(),
        'gen_optimizer': self.gen_optimizer.state_dict(),
        'disc_optimizer': self.disc_optimizer.state_dict()
    }

  def load_state_dict(self, state_dict):
    self.gen.load_state_dict(state_dict['generator'])

    if self.disc is not None:
      assert 'discriminator' in state_dict, 'Incompatible checkpoint'
      self.disc.load_state_dict(state_dict['discriminator'])

    if self.gen_optimizer is not None:
      assert 'gen_optimizer' in state_dict, 'Incompatible checkpoint'
      self.gen_optimizer.load_state_dict(state_dict['gen_optimizer'])

    if self.disc_optimizer is not None:
      assert 'disc_optimizer' in state_dict, 'Incompatible checkpoint'
      self.disc_optimizer.load_state_dict(state_dict['disc_optimizer'])

  def __str__(self):
    s = 'Generator:\n'
    s += str(self.gen)
    if self.disc is not None:
      s += '\nDiscriminator:\n'
      s += str(self.disc)
    return s

  def epoch_beginning(self, epoch):
    if is_pre_epoch_scheduler(self.gen_lr_scheduler):
      self.gen_lr_scheduler.step()
    if is_pre_epoch_scheduler(self.disc_lr_scheduler):
      self.disc_lr_scheduler.step()

  def epoch_finished(self, epoch):
    if is_post_epoch_scheduler(self.gen_lr_scheduler):
      self.gen_lr_scheduler.step()
    if is_post_epoch_scheduler(self.disc_lr_scheduler):
      self.disc_lr_scheduler.step()

  @staticmethod
  def _update_step(optimizer, losses, weights):
    total_loss = torch.sum(torch.cat(losses) * weights)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    return total_loss

  def _train_step(self, loader):
    batch = self._request_data(loader)
    if batch is None:
      return 0, None, None

    inp, target = batch

    # Propagate fake image through discriminator
    out_gen = self.gen(inp)
    out_disc_fake = self.disc(self.disc_input_fn(out_gen, inp, detach=True))

    # Propagate real images through discriminator
    out_disc_real = self.disc(self.disc_input_fn(target, inp, detach=True))

    loss_metrics = {}
    disc_losses = []
    # Compute discriminator losses
    for name, criterion in self.disc_adv_criteria.items():
      loss = criterion(out_disc_fake, out_disc_real)
      disc_losses.append(loss)
      loss_metrics['disc_loss_' + name] = get_loss_metric(loss.data[0])

    # Propagate again with non-detached input to allow gradients on the
    # generator
    out_disc_fake = self.disc(self.disc_input_fn(out_gen, inp, detach=False))

    # Compute adversarial generator losses from discriminator output
    # Order matters: first compute adversarial losses for generator, then
    # the other generator losses. Otherwise the loss weights will not match
    gen_losses = []
    for name, criterion in self.gen_adv_criteria.items():
      loss = criterion(out_disc_fake)
      gen_losses.append(loss)
      loss_metrics['gen_loss_' + name] = get_loss_metric(loss.data[0])

    # Compute generator losses on prediction and target image
    for name, criterion in self.gen_criteria.items():
      loss = criterion(out_gen, target)
      gen_losses.append(loss)
      loss_metrics['gen_loss_' + name] = get_loss_metric(loss.data[0])

    # Perform updates
    total_disc_loss = self._update_step(self.disc_optimizer,
                                        disc_losses,
                                        self.disc_loss_weights)
    total_gen_loss = self._update_step(self.gen_optimizer,
                                       gen_losses,
                                       self.gen_loss_weights)

    loss_metrics['disc_loss'] = get_loss_metric(total_disc_loss.data[0])
    loss_metrics['gen_loss'] = get_loss_metric(total_gen_loss.data[0])

    return 1, loss_metrics, (inp, out_gen, target, out_disc_fake)

  def _val_step(self, loader, compute_metrics=True):
    batch = self._request_data(loader, volatile=True)
    if batch is None:
      return None, None

    inp, target = batch
    prediction = self.gen(inp)

    loss_metrics = {}
    if compute_metrics:
      # Only compute the standard losses here, adversarial losses don't make
      # to much sense
      for name, criterion in self.gen_criteria.items():
        loss = criterion(prediction, target)
        loss_metrics['gen_loss_' + name] = get_loss_metric(loss.data[0])

    return loss_metrics, (inp, prediction, target, None)

  def _compute_gen_metrics(self, metrics, metric_fns, metric_transform,
                           predictions, targets):
    if metric_transform is not None:
      predictions, targets = metric_transform(predictions, targets)

    for metric_name, metric_fn in metric_fns.items():
      metric = metric_fn(predictions, targets)
      metrics['gen_' + metric_name] = metric

    return metrics

  def _compute_disc_metrics(self, metrics, metric_fns, out_disc_fake):
    prob = out_disc_fake['prob']
    if self.cuda != '':
        target = torch.cuda.ByteTensor(prob.shape).fill_(0)
    else:
        target = torch.ByteTensor(prob.shape).fill_(0)

    for metric_name, metric_fn in metric_fns.items():
      metric = metric_fn(prob, target)
      metrics['disc_' + metric_name] = metric

    return metrics

  def _compute_train_metrics(self, data):
    metrics = {}
    gen_metrics = self._compute_gen_metrics(metrics,
                                            self.train_gen_metric_fns,
                                            self.train_gen_metric_transform,
                                            data[1], data[2])
    disc_metrics = self._compute_disc_metrics(metrics,
                                              self.train_disc_metric_fns,
                                              data[3])
    return metrics

  def _compute_test_metrics(self, data):
    return self._compute_gen_metrics({},
                                     self.val_metric_fns,
                                     self.val_metric_transform,
                                     data[1], data[2])

  def _set_train(self):
    self.gen.train()
    self.disc.train()

  def _set_test(self):
    self.gen.eval()
    if self.disc is not None:
      self.disc.eval()
