import torch.nn as nn

from models.criteria import get_criterion
from utils.tensor_transforms import normalize_range


class VGGLoss(nn.Module):
  def __init__(self, loss_name, cuda, blocks=-1, criterion='L1', weights=None):
    super(VGGLoss, self).__init__()

    if loss_name == 'VGG19':
      from models.vgg import VGG19
      module = VGG19
    else:
      raise ValueError('Unknown VGG loss {}'.format(loss_name))

    if blocks == -1:
      blocks = [module.LAST_FEATURE_MAP]
    elif not isinstance(blocks, list):
      blocks = [blocks]

    self.model = module(blocks, requires_grad=False)

    if cuda != '':
      self.model = self.model.cuda()

    self.criterion = get_criterion(None, criterion)
    if weights is not None:
      assert len(weights) == len(blocks)
      self.weights = weights
    else:
      self.weights = [1.] * len(blocks)

  def forward(self, prediction, target):
    # We assume here that the input is in range (-1, 1)
    prediction = normalize_range(prediction, source_range=(-1., 1.))
    target = normalize_range(target.detach(), source_range=(-1., 1.))

    pred_features = self.model(prediction)
    target_features = self.model(target)

    loss = 0
    for weight, pred_feature, target_feature in zip(self.weights,
                                                    pred_features,
                                                    target_features):
      loss += weight * self.criterion(pred_feature, target_feature.detach())

    return loss
