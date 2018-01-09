import os

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')


def _is_image_file(filename):
  _, ext = os.path.splitext(filename)
  return ext.lower() in IMAGE_EXTENSIONS


def _load_image(path, convert_to_grayscale=False, convert_to_luma=False):
  img = Image.open(path)

  if convert_to_grayscale:
    if img.mode != 'L':
      img = img.convert('L')
  elif convert_to_luma:
    img = img.convert('YCbCr').split()[0]
  else:
    if img.mode == 'L':  # For random grayscale images in RGB datasets
      img = img.convert('RGB')

  return img


class SRDatasetFromImagePaths(Dataset):
  """Loader for super resolution where the input is used as target"""
  def __init__(self, images, input_images=None, grayscale=False,
               luma=False, transform=None, name=None, no_target=False,
               upscale_factor=None):
    super(SRDatasetFromImagePaths, self).__init__()
    assert input_images is None or len(images) == len(input_images), \
        'Number of input images need to match number of images'
    if no_target:
       assert upscale_factor is not None
    self.name = name
    self.images = images
    self.input_images = input_images
    self.grayscale = grayscale
    self.luma = luma
    self.transform = transform
    self.no_target = no_target
    self.upscale_factor = upscale_factor

  def __getitem__(self, index):
    image = _load_image(self.images[index],
                        convert_to_grayscale=self.grayscale,
                        convert_to_luma=self.luma)
    if self.input_images is not None:
      inp = _load_image(self.input_images[index],
                        convert_to_grayscale=self.grayscale,
                        convert_to_luma=self.luma)
    else:
      inp = image.copy()

    if not self.no_target:
      target = image
    else:
      # For purely super-resolving images, there is no target image.
      # As our pipeline requires one, we generate one with a fitting size
      size = (int(inp.size[0] * self.upscale_factor),
              int(inp.size[1] * self.upscale_factor))
      target = Image.new(inp.mode, size)

    if self.transform:
      inp, target = self.transform(inp, target)

    return inp, target

  def __len__(self):
    return len(self.images)


def make_sr_dataset_from_folder(conf, file_or_dir, transform,
                                inference=False, name=None):
  if os.path.isdir(file_or_dir):
    images = [os.path.join(file_or_dir, filename)
              for filename in sorted(os.listdir(file_or_dir))
              if _is_image_file(filename)]
  elif os.path.isfile(file_or_dir):
    assert _is_image_file(file_or_dir), \
        '{} is not image file'.format(file_or_dir)
    images = [file_or_dir]
  else:
    assert False, '{} does not exist'.format(file_or_dir)

  assert len(images) > 0, 'No image files found for {}'.format(file_or_dir)

  if name is None:
    name = os.path.basename(file_or_dir)

  return SRDatasetFromImagePaths(images,
                                 grayscale=conf.get_attr('grayscale',
                                                         default=False),
                                 luma=conf.get_attr('luma', default=False),
                                 transform=transform,
                                 name=name,
                                 no_target=inference,
                                 upscale_factor=conf.upscale_factor)
