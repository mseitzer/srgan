import os

from data.datasets import SRDatasetFromImagePaths
from data.transform_wrappers import get_sr_transform

DATASET_DIR = 'Set5'

NUM_IMAGES = 5
_FMT_IMAGE_NAMES = 'image_SRF_{scale}/img_{index:03d}_SRF_{scale}_{mode}.png'


def _get_images(dataset_path, scale, mode):
  assert 2 <= scale <= 4
  assert mode == 'HR' or mode == 'LR'
  scale = int(scale)
  image_names = (_FMT_IMAGE_NAMES.format(scale=scale, index=idx, mode=mode)
                 for idx in range(1, NUM_IMAGES + 1))
  images = [os.path.join(dataset_path, image_name)
            for image_name in image_names]
  for image in images:
    assert os.path.exists(image), 'Image {} does not exist'.format(image)

  return images


def get_train_set(conf, data_dir):
  """Get training dataset

  Not implemented
  """
  raise Exception('Set5 has no training set')


def _get_test_or_val_set(conf, data_dir):
  """Gets validation or training dataset

  Each entry consists of input, the low resolution center crop of an image, and
  target, the center crop in high resolution. If `full_image` is set to true
  in conf, then the full image is used instead of a crop.

  Parameters
  ----------
  conf : Configuration
    Configuration expecting the following keys:
      - `test_crop_size`: Size of the crop to extract
      - `upscale_factor`: Downscale the crop by this factor
      - `scale_to_orig`: If true, scale the crop up to crop size again after
        downscaling to preserve same size between inputs and targets
      - `grayscale`: If true, converts images to grayscale
  data_dir : string
    Path to top level data folder

  Returns : torch.utils.data.Dataset
  -------
    Instance of validation or test dataset
  """
  dataset_path = os.path.join(data_dir, DATASET_DIR)

  hr_images = _get_images(dataset_path, scale=2, mode='HR')

  if 2 <= conf.upscale_factor <= 4:
    # Use downscaled images from dataset, if available
    sr_images = _get_images(dataset_path, scale=conf.upscale_factor, mode='LR')
    transform = get_sr_transform(conf, mode='test', downscale=False)
  else:
    # Need to downscale images ourselves
    sr_images = None
    transform = get_sr_transform(conf, mode='test', downscale=True)

  dataset = SRDatasetFromImagePaths(hr_images, sr_images,
                                    grayscale=conf.get_attr('grayscale',
                                                            default=False),
                                    luma=conf.get_attr('luma', default=False),
                                    transform=transform,
                                    name=DATASET_DIR)
  return dataset


def get_val_set(conf, data_dir):
  """Get validation dataset

  Parameters: see _get_test_or_val_set
  """
  return _get_test_or_val_set(conf, data_dir)


def get_test_set(conf, data_dir):
  """Get test dataset

  Parameters: see _get_test_or_val_set
  """
  return _get_test_or_val_set(conf, data_dir)
