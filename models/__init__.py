import importlib

MODEL_MODULES = {
    'bilinear': 'models.bilinear',
    'SRResNet': 'models.srresnet',
    'CNNDiscriminator': 'models.discriminators'
}


def construct_model(conf, model_name):
  assert model_name in MODEL_MODULES, \
      'Unknown model {}'.format(model_name)

  module = importlib.import_module(MODEL_MODULES[model_name])
  model = module.construct_model(conf, model_name)
  return model
