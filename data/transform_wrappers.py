from PIL import Image


def _build_param_dict(conf, required_params, optional_params=[],
                      key_renames={}, kwargs={}):
  # Filter out params which are passed in kwargs
  required_params = [p for p in required_params if p not in kwargs]
  param_dict = conf.to_param_dict(required_params,
                                  optional_params.copy(),
                                  key_renames)
  param_dict.update(kwargs)
  return param_dict


def get_sr_transform(conf, mode, **kwargs):
  def _get_interp(method):
    if method == 'bilinear':
      return Image.BILINEAR
    elif method == 'bicubic':
      return Image.BICUBIC
    else:
      raise ValueError('Unknown interpolation method {}'.format(method))

  assert mode in ('train', 'test')

  required_params = [
      'train_crop_size', 'upscale_factor'
  ]

  # Maps keys in configuration to parameter names in transform functions
  key_renames = {
      'train_crop_size': 'crop_size',
      'test_crop_size': 'crop_size',
      'scale_to_orig': 'upscale'
  }

  if mode == 'train':
    from data.sr_transforms import sr_train_transform
    optional_params = {
        'scale_to_orig': False,
        'interpolation': 'bilinear'
    }

    param_dict = _build_param_dict(conf,
                                   required_params,
                                   optional_params,
                                   key_renames,
                                   kwargs)
    param_dict['interpolation'] = _get_interp(param_dict['interpolation'])
    transform = sr_train_transform(**param_dict)
  else:
    from data.sr_transforms import sr_test_transform
    optional_params = {
        'scale_to_orig': False,
        'interpolation': 'bilinear',
        'full_image': False
    }
    param_dict = _build_param_dict(conf,
                                   required_params,
                                   optional_params,
                                   key_renames,
                                   kwargs)
    param_dict['interpolation'] = _get_interp(param_dict['interpolation'])
    transform = sr_test_transform(**param_dict)

  return transform


def get_sr_output_transform(conf, mode, **kwargs):
  assert mode in ('train', 'test', 'output')
  if mode == 'train':
    from data.sr_transforms import sr_output_train_transform
    transform = sr_output_train_transform(convert_luma=True)
  else:
    from data.sr_transforms import sr_output_test_transform
    if mode == 'test':
      param_dict = _build_param_dict(conf,
                                     ['upscale_factor'],
                                     kwargs={'crop_border_pixels': True,
                                             'convert_luma': True})
    else:
      param_dict = _build_param_dict(conf,
                                     ['upscale_factor'])
    transform = sr_output_test_transform(**param_dict)

  return transform


def get_output_transform(conf, application, mode, **kwargs):
  applications = {
      'super_resolution': get_sr_output_transform,
  }

  assert application in applications
  return applications[application](conf, mode, **kwargs)
