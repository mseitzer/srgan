import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import (Compose, Resize, ToTensor,
                                    CenterCrop, Lambda)

from data.transform_utils import get_random_crop_params
from utils.tensor_transforms import (normalize_range,
                                     scale_to_range,
                                     convert_to_luma)


def _crop_border_pixels(img, scale, min_pixels_to_ignore=0):
  px = int(scale) + min_pixels_to_ignore
  return img[:, :, px:-px, px:-px]


def _maybe_pad_image_to_size(image, target_size):
  tw, th = target_size
  w, h = image.size
  if w < tw or h < th:
    padded_image = Image.new(image.mode, target_size)
    padded_image.paste(image, ((tw - w) // 2, (th - h) // 2))
    image = padded_image

  return image


def _maybe_pad_to_target_size(prediction, target, mode='reflect'):
  def get_padding(pred_size, target_size, fixed_bottom_padding=1):
    """Computes padding needed to pad image from one size to another

    Padding gets equally distributed on both sides, with the right/bottom side
    having one more pixel if the needed padding is uneven. Additionally, the
    method supports always giving the right/bottom side a fixed amount of
    padding from the total padding (e.g. get_padding(pred_size=1, target_size=5,
    fixed_bottom_padding=1) = (1, 3)). This is needed to exactly align output
    images where the target image's size is not divisible by the scaling factor.

    Parameters
    ----------
    pred_size : int
      Size of the image in x or y direction
    target_size : int
      Size the image has after padding in x or y direction
    fixed_bottom_padding : int
      Number of pixels always added to right or bottom side from the total
      number of padded pixels

    Returns
    -------
    Tuple (a, b), where a is the padding to be applied on the left/top side, and
    b is the padding to be applied on the right/bottom side
    """
    total_padding = target_size - pred_size
    if total_padding == 0:
      return 0, 0
    total_padding -= fixed_bottom_padding
    padding = total_padding // 2
    if total_padding % 2 == 0:
      return padding, padding + fixed_bottom_padding
    else:
      return padding, padding + fixed_bottom_padding + 1

  pred_size = prediction.size()
  target_size = target.size()
  if pred_size != target_size:
    padding_top, padding_bottom = get_padding(pred_size[2], target_size[2])
    padding_left, padding_right = get_padding(pred_size[3], target_size[3])
    return F.pad(prediction,
                 (padding_left, padding_right, padding_top, padding_bottom),
                 mode=mode)
  else:
    return prediction


def _get_rounded_size(size, scale_factor):
  if isinstance(size, (int, float)):
    return int(size - size % scale_factor)
  else:
    return [int(s - s % scale_factor) for s in size]


def _get_downscaled_size(size, downscale_factor):
  if isinstance(size, (int, float)):
    return int(_get_rounded_size(size, downscale_factor) // downscale_factor)
  else:
    size = _get_rounded_size(size, downscale_factor)
    return [int(s // downscale_factor) for s in size]


def _crop(image, crop_params):
  i, j, h, w = crop_params
  return image.crop((j, i, j + w, i + h))


def sr_train_transform(crop_size, upscale_factor, downscale=True,
                       upscale=False, interpolation=Image.BILINEAR):
  def transform(inp, target):
    inp_size = crop_size if downscale else lr_size

    inp_crop = get_random_crop_params(inp.size, inp_size)
    target_crop = (inp_crop[0], inp_crop[1], crop_size[0], crop_size[1])

    inp = _crop(inp, inp_crop)
    inp = _maybe_pad_image_to_size(inp, inp_size)

    target = _crop(target, target_crop)
    target = _maybe_pad_image_to_size(target, crop_size)

    return input_transform(inp), target_transform(target)

  if isinstance(crop_size, int):
    crop_size = (crop_size, crop_size)

  input_transforms = []
  lr_size = _get_downscaled_size(crop_size, upscale_factor)
  if downscale:
    # Input is HR image, need to downscale
    input_transforms.append(Resize(lr_size, interpolation=interpolation))
  if upscale:
    input_transforms.append(Resize(crop_size, interpolation=interpolation))
  input_transforms.append(ToTensor())

  input_transform = Compose(input_transforms)
  target_transform = Compose([ToTensor(),
                              Lambda(lambda t: scale_to_range(t, (-1., 1.)))])

  return transform


def sr_test_transform(crop_size, upscale_factor, downscale=True, upscale=False,
                      full_image=False, interpolation=Image.BILINEAR):
  def transform(inp, target):
    return input_transform(inp), target_transform(target)

  def adaptive_scale(img):
    if downscale:
      # Input is HR image, need to downscale
      hr_size = _get_rounded_size(img.size, upscale_factor)
      lr_size = _get_downscaled_size(img.size, upscale_factor)
      img = img.crop((0, 0, hr_size[0], hr_size[1]))
      img = img.resize(lr_size, interpolation)
    else:
      # Input already is LR image
      hr_size = _get_downscaled_size(img.size, 1. / upscale_factor)
    if upscale:
      img = img.resize(hr_size, interpolation)
    return img

  if full_image:
    input_transforms = [Lambda(adaptive_scale)]
  else:
    lr_size = _get_downscaled_size(crop_size, upscale_factor)
    if downscale:
      # Input is HR image, need to downscale
      input_transforms = [CenterCrop(crop_size)]
      input_transforms.append(Resize(lr_size, interpolation=interpolation))
    else:
      # Input already is LR image, take smaller crop
      input_transforms = [CenterCrop(lr_size)]
    if upscale:
      input_transforms.append(Resize(crop_size, interpolation=interpolation))

  input_transforms.append(ToTensor())
  input_transform = Compose(input_transforms)

  target_transforms = []
  if not full_image:
    target_transforms.append(CenterCrop(crop_size))
  target_transforms += [ToTensor(),
                        Lambda(lambda t: scale_to_range(t, (-1., 1.)))]
  target_transform = Compose(target_transforms)

  return transform


def sr_output_train_transform(convert_luma):
  def transform(pred, target):
    return pred_transform(pred), target_transform(target)

  transforms = [Lambda(lambda img: normalize_range(img, (-1., 1.)))]
  if convert_luma:
    transforms.append(Lambda(lambda img: convert_to_luma(img)))

  pred_transform = Compose(transforms)
  target_transform = Compose(transforms)
  return transform


def sr_output_test_transform(upscale_factor,
                             crop_border_pixels=False,
                             convert_luma=False):
  def transform(pred, target):
    pred = _maybe_pad_to_target_size(pred, target)
    return pred_transform(pred), target_transform(target)

  transforms = []
  if crop_border_pixels:
    transforms.append(Lambda(lambda img: _crop_border_pixels(img,
                                                             upscale_factor)))
  transforms.append(Lambda(lambda img: normalize_range(img, (-1., 1.))))
  if convert_luma:
    transforms.append(Lambda(lambda img: convert_to_luma(img)))

  pred_transform = Compose(transforms)
  target_transform = Compose(transforms)
  return transform
