import os

from data.datasets import make_sr_dataset_from_folder
from data.transform_wrappers import get_sr_transform

DATASET_DIR = 'COCO'
TRAIN_DIR = 'images/train2017'
VAL_DIR = 'images/val2017'
TEST_DIR = 'images/test2017'


def get_train_set(conf, data_dir):
  """Get training dataset

  Each entry consists of input, a random low resolution random crop, and
  target, the random crop in high resolution.

  Parameters
  ----------
  conf : Configuration
    Configuration expecting the following keys:
      - `train_crop_size`: Size of the random crop to extract
      - `upscale_factor`: Downscale the random crop by this factor
      - `scale_to_orig`: If true, scale random crop up to crop size again after
        downscaling to preserve same size between inputs and targets
      - `grayscale`: If true, converts images to grayscale
  data_dir : string
    Path to top level data folder

  Returns : torch.utils.data.Dataset
  -------
    Instance of training dataset
  """
  train_dir = os.path.join(data_dir, DATASET_DIR, TRAIN_DIR)
  transform = get_sr_transform(conf, mode='train')
  return make_sr_dataset_from_folder(conf, train_dir, transform,
                                     name=DATASET_DIR)


def _get_test_or_val_set(conf, data_dir, fold_dir):
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
  fold_dir : string
    Subdirectory of the fold to use

  Returns : torch.utils.data.Dataset
  -------
    Instance of validation or test dataset
  """
  fold_dir = os.path.join(data_dir, DATASET_DIR, fold_dir)
  transform = get_sr_transform(conf, mode='test')
  return make_sr_dataset_from_folder(conf, fold_dir, transform,
                                     name=DATASET_DIR)


def get_val_set(conf, data_dir):
  """Get validation dataset

  Parameters: see _get_test_or_val_set
  """
  return _get_test_or_val_set(conf, data_dir, VAL_DIR)


def get_test_set(conf, data_dir):
  """Get test dataset

  Parameters: see _get_test_or_val_set
  """
  return _get_test_or_val_set(conf, data_dir, TEST_DIR)
