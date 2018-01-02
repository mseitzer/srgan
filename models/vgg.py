import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision.transforms import Normalize


class VGG19(nn.Module):
  """Pretrained VGG19 network, without fully connected layers"""

  LAST_FEATURE_MAP = 4  # Index of default block of VGG19 to return

  def __init__(self, output_blocks=[LAST_FEATURE_MAP], requires_grad=False):
    """Build pretrained VGG19

    Parameters
    ----------
    output_blocks : list of int
      Indices of feature blocks to return. Each feature block ends
      before a max-pooling layer, i.e. the output of a feature block is the
      output of the last convolutional layer and activation right before the
      feature map is downscaled. The last possible block index is 5, a block
      which only consists of the last max-pooling in the network
    requires_grad : bool
      If true, parameters of the model require gradient. Possibly useful for
      finetuning the network
    """
    super(VGG19, self).__init__()
    assert len(output_blocks) >= 1, 'Need at least one output block'
    self.output_blocks = sorted(output_blocks)
    last_needed_block = self.output_blocks[-1]

    assert last_needed_block <= 5, 'VGG19 has at most 6 blocks'

    layers = models.vgg19(pretrained=True).features
    self.blocks = nn.ModuleList([nn.Sequential()])
    for idx, layer in enumerate(layers):
      if isinstance(layer, nn.MaxPool2d):
        if len(self.blocks) - 1 == last_needed_block:
          break
        self.blocks.append(nn.Sequential())

      self.blocks[-1].add_module(str(idx), layer)

    for param in self.parameters():
      param.requires_grad = requires_grad

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    mean_tensor = torch.from_numpy(mean.reshape(1, 3, 1, 1))
    self.register_buffer('mean', Variable(mean_tensor, requires_grad=False))

    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    std_tensor = torch.from_numpy(std.reshape(1, 3, 1, 1))
    self.register_buffer('std', Variable(std_tensor, requires_grad=False))

  def forward(self, x):
    """Get VGG19 feature maps

    Parameters
    ----------
    x : torch.Tensor
      Input tensor of shape BxCxHxW. Values are expected to be in range (0, 1)
    """
    # Normalize input to the statistics VGG19 expects
    x = x.sub(self.mean).div(self.std)

    out = []
    for block_idx, block in enumerate(self.blocks):
      x = block(x)
      if block_idx in self.output_blocks:
        out.append(x)

    return out
