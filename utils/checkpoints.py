import os

import torch


def save_checkpoint(log_file_path, conf, runner, epoch, best_val_metrics):
  state = {
      'conf': conf,
      'runner': runner.state_dict(),
      'epoch': epoch,
      'best_val_metrics': best_val_metrics
  }
  torch.save(state, log_file_path)


def restore_checkpoint(checkpoint_path, runner, cuda=None):
  # This handles restoring weights on the CPU if needed
  map_location = lambda storage, loc: storage if cuda == '' else None

  checkpoint = torch.load(checkpoint_path, map_location=map_location)

  runner.load_state_dict(checkpoint['runner'])

  state = {
      'conf': checkpoint['conf'],
  }

  if 'epoch' in checkpoint:
    state['start_epoch'] = checkpoint['epoch']
  if 'best_val_metrics' in checkpoint:
    state['best_val_metrics'] = checkpoint['best_val_metrics']

  return state


def inference_checkpoint_from_training_checkpoint(checkpoint, runner_type):
  inference_net_by_runner_type = {
      'standard': 'model',
      'adversarial': 'generator',
  }
  assert runner_type in inference_net_by_runner_type, \
      'Unknown runner_type {}'.format(runner_type)

  inference_net = inference_net_by_runner_type[runner_type]
  assert inference_net in checkpoint['runner'], \
      'Checkpoint does not support runner_type {}'.format(runner_type)

  state = {
      'conf': checkpoint['conf'],
      'runner': {}
  }

  state['runner'][inference_net] = checkpoint['runner'][inference_net]
  return state


def prune_checkpoints(run_dir, num_checkpoints_to_retain=1):
  from utils.checkpoint_paths import is_checkpoint_path
  checkpoints = [f for f in os.listdir(run_dir) if is_checkpoint_path(f)]
  num_checkpoints = len(checkpoints)
  if num_checkpoints > num_checkpoints_to_retain:
    for f in sorted(checkpoints)[:num_checkpoints - num_checkpoints_to_retain]:
      chkpt_path = os.path.join(run_dir, f)
      try:
        os.remove(chkpt_path)
      except OSError:
        print('Could not remove old checkpoint {}'.format(chkpt_path))


def load_model_state_dict(checkpoint_path, model_key, cuda):
   # This handles restoring weights on the CPU if needed
   map_location = lambda storage, loc: storage if cuda == '' else None

   checkpoint = torch.load(checkpoint_path, map_location=map_location)

   if 'runner' not in checkpoint:
     raise ValueError(('Did not find runner in checkpoint {}. '
                       'Old checkpoint?').format(checkpoint_path))

   runner_state = checkpoint['runner']
   if model_key not in runner_state:
     raise ValueErorr(('Did not find model {} '
                       'in checkpoint {}').format(model_key, checkpoint_path))

   return runner_state[model_key]

