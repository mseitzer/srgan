import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

_LOSS_TYPE_GENERATOR = 0
_LOSS_TYPE_DISCRIMINATOR = 1


class AdversarialLoss(nn.Module):
  def __init__(self, loss_name, loss_type, cuda):
    super(AdversarialLoss, self).__init__()
    losses = {
        'gan': (self.gan_loss_gen, self.gan_loss_disc)
    }
    assert loss_type == 'disc' or loss_type == 'gen'
    assert loss_name in losses, 'Unknown loss {}'.format(loss_name)

    if loss_type == 'disc':
      self.loss_type = _LOSS_TYPE_DISCRIMINATOR
    else:
      self.loss_type = _LOSS_TYPE_GENERATOR

    self.real_label = 1.0
    self.fake_label = 0.0
    self.real_label_var = None
    self.fake_label_var = None

    if cuda != '':
      self.tensor_fn = lambda *args: torch.FloatTensor(*args).cuda()
    else:
      self.tensor_fn = lambda *args: torch.FloatTensor(*args)

    # Set actual function to use
    self.forward = losses[loss_name][self.loss_type]

  def _get_label_var(self, prob, is_real):
    if is_real:
      if self.real_label_var is None or \
         self.real_label_var.numel() != prob.numel():
        tensor = self.tensor_fn(prob.shape).fill_(self.real_label)
        self.real_label_var = Variable(tensor, requires_grad=False)
      return self.real_label_var
    else:
      if self.fake_label_var is None or \
         self.fake_label_var.numel() != prob.numel():
        tensor = self.tensor_fn(prob.shape).fill_(self.fake_label)
        self.fake_label_var = Variable(tensor, requires_grad=False)
      return self.fake_label_var

  def gan_loss_disc(self, out_disc_fake, out_disc_real):
    prob_fake = out_disc_fake['prob']
    prob_real = out_disc_real['prob']

    fake_label = self._get_label_var(prob_fake, is_real=False)
    loss_fake = F.binary_cross_entropy(prob_fake, fake_label)

    real_label = self._get_label_var(prob_real, is_real=True)
    loss_real = F.binary_cross_entropy(prob_real, real_label)

    return loss_fake + loss_real

  def gan_loss_gen(self, out_disc_fake):
    prob_fake = out_disc_fake['prob']
    real_label = self._get_label_var(prob_fake, is_real=True)
    loss_fake = F.binary_cross_entropy(prob_fake, real_label)
    return loss_fake

