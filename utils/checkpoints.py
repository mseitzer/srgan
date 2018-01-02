import os

import torch

from utils.checkpoint_paths import is_checkpoint_path


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

  if 'runner' in checkpoint:
    runner.load_state_dict(checkpoint['runner'])
  else:
    # Backwards compatibility
    runner.load_state_dict({'model': checkpoint['model'],
                            'optimizer': checkpoint['optimizer']})

  return {
      'conf': checkpoint['conf'],
      'start_epoch': checkpoint['epoch'],
      'best_val_metrics': checkpoint['best_val_metrics']
  }


def prune_checkpoints(run_dir, num_checkpoints_to_retain=1):
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

