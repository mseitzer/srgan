import torch


def normalize_range(tensor, source_range, clamp=True):
  """Scales tensor from `source_range` to (0, 1) range"""
  tensor = (tensor - source_range[0]) / (source_range[1] - source_range[0])
  if clamp:
      tensor = tensor.clamp(source_range[0], source_range[1])
  return tensor


def scale_to_range(tensor, target_range, clamp=True):
  """Scales tensor from (0, 1) range to `target range`"""
  tensor = tensor * (target_range[1] - target_range[0]) + target_range[0]
  if clamp:
    tensor = tensor.clamp(target_range[0], target_range[1])
  return tensor


def convert_to_luma(tensor, use_digital_rgb=False):
  """Convert RGB tensor to luma channel

  This functions translates an RGB image to a luma image
  with range (16/255, 235/255). This corresponds to Matlabs rgb2ycbcr
  function, or the transform ITU-R BT.601 analog RGB to digital YCbCr
  (see https://en.wikipedia.org/wiki/YCbCr).
  This is different from using PIL's Image.convert('YCbCr') function, which
  converts to JPEG YCbCr and has a target range of (0, 1).

  Most super resolution papers *should* use the Matlab version for
  evaluation, although it is not exactly clear.

  This function also supports the ITU-R BT.601 digital rgb to digital YCbCr
  conversion by passing `use_digitial_rgb=True`.

  Parameters
  ----------
  tensor : torch.Tensor
    Tensor of shape (batch, 3, height, width)
  use_digital_rgb : bool
    Assume digital RGB input
  Returns
  -------
    Tensor of shape (batch, 1, height, width)
  """
  assert tensor.dim() == 4 and tensor.size()[1] == 3

  if use_digital_rgb:
    scale = [65.481, 128.553, 24.966]
  else:
    scale = [65.738, 129.057, 25.064]

  luma = (scale[0] * tensor[:, 0, :, :] +
          scale[1] * tensor[:, 1, :, :] +
          scale[2] * tensor[:, 2, :, :] + 16.)
  luma = luma.clamp(16., 235.) / 255.
  return luma.unsqueeze(dim=1)
