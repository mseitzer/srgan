import importlib

RUNNER_MODULES = {
    'standard': 'training.runner',
    'adversarial': 'training.adversarial_runner'
}


def build_runner(conf, runner_type, cuda, mode='train', resume=False):
  assert runner_type in RUNNER_MODULES, \
      'Unknown runner {}'.format(runner_type)

  module = importlib.import_module(RUNNER_MODULES[runner_type])
  runner = module.build_runner(conf, cuda, mode, resume)
  return runner
