import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def compute_psnr(prediction, target):
  """Calculates peak signal-to-noise ratio between target and prediction

  Parameters
  ----------
  prediction : torch.Tensor or torch.autograd.Variable
    Predicted image
  target : torch.Tensor or torch.autograd.Variable
    Target image
  """
  mse = F.mse_loss(prediction, target).data[0]
  psnr = 10. * np.log10(1. / mse)
  return psnr


def compute_ssim(prediction, target, window_size=11):
  """Calculates structural similarity index between target and prediction

  Parameters
  ----------
  prediction : torch.Tensor or torch.autograd.Variable
    Predicted image
  target : torch.Tensor or torch.autograd.Variable
    Target image
  window_size : int
    Size of the Gaussian kernel used for computing SSIM
  """
  from metrics import pytorch_ssim

  if not isinstance(prediction, Variable):
    prediction = Variable(prediction, volatile=True)
  if not isinstance(target, Variable):
    target = Variable(target, volatile=True)

  ssim = pytorch_ssim.ssim(prediction, target, window_size=window_size).data[0]
  return ssim
