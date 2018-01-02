import random


def get_random_crop_params(orig_size, target_size):
  """Gets bounding box parameters for a random crop

  Parameters
  ----------
  orig_size : int or tuple
    Size of image in the form of (width, height)
  target_size : int or tuple
    Size of random crop in the form of (width, height)

  Returns
  -------
    Bounding box of random crop in the form of (top, left, height, width)
  """
  if isinstance(orig_size, int):
    orig_size = (orig_size, orig_size)
  if isinstance(target_size, int):
    target_size = (target_size, target_size)

  w, h = orig_size
  tw, th = target_size
  if w <= tw and h <= th:
    return 0, 0, h, w

  i = random.randint(0, h - th)
  j = random.randint(0, w - tw)
  return i, j, th, tw
