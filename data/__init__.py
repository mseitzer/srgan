import importlib

_DATASET_MODULES = {
    'BSDS500': 'data.bsds500',
    'Set5': 'data.set5',
    'Set14': 'data.set14',
    'COCO': 'data.coco',
    'ScarSeg': 'data.reconstruction.scar_segmentation'
}


def is_dataset(dataset_name):
  return dataset_name in _DATASET_MODULES


def load_dataset(conf, data_dir, dataset_name, fold):
  """Load dataset

  Parameters
  ----------
  conf : Configuration
    Configuration to pass to the dataset loader
  data_dir : string
    Path to top level data folder
  dataset_name : string
    Dataset name
  fold : string
    Either `train`, `val`, or `test` fold
  """
  assert fold in ('train', 'val', 'test')
  assert dataset_name in _DATASET_MODULES, \
      'Unknown dataset {}'.format(dataset_name)

  module = importlib.import_module(_DATASET_MODULES[dataset_name])

  if fold == 'train':
    return module.get_train_set(conf, data_dir)
  elif fold == 'val':
    return module.get_val_set(conf, data_dir)
  elif fold == 'test':
    return module.get_test_set(conf, data_dir)

  return None
