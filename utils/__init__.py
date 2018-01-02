import os


def _get_free_gpu_idx():
  from utils.gpu_observer import run_nvidiasmi_local, get_gpu_infos
  gpu_infos = get_gpu_infos(run_nvidiasmi_local())
  free_gpus = [info for info in gpu_infos if len(info['pids']) == 0]
  return None if len(free_gpus) == 0 else free_gpus[0]['idx']


def set_cuda_env(gpu_idx):
  """Sets CUDA_VISIBLE_DEVICES environment variable

  Parameters
  ----------
  gpu_idx : string
    Index of GPU to use, `auto`, or empty string . If `auto`, attempts to
    automatically select a free GPU.

  Returns
  -------
    Value environment variable has been set to

  Raises
  ------
    Exception if auto selecting GPU has been attempted, but failed
  """
  if gpu_idx == 'auto':
    gpu_idx = _get_free_gpu_idx()
    if gpu_idx is None:
      raise Exception('No free GPU on this machine. Aborting run.')
    gpu_idx = str(gpu_idx)

  os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx
  return gpu_idx


def set_random_seeds(seed):
  import random
  import numpy as np
  import torch
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def cpuify(modules_or_tensors):
  if isinstance(modules_or_tensors, dict):
    return {key: cpuify(values) for key, values in modules_or_tensors.items()}
  elif isinstance(modules_or_tensors, tuple):
    modules_or_tensors = list(modules_or_tensors)

  if isinstance(modules_or_tensors, list):
    for idx, obj in enumerate(modules_or_tensors):
      if obj is not None:
        modules_or_tensors[idx] = obj.cpu()
    return modules_or_tensors
  return modules_or_tensors.cpu()


def cudaify(modules_or_tensors, device_id=None):
  if isinstance(modules_or_tensors, dict):
    return {key: cudaify(values, device_id)
            for key, values in modules_or_tensors.items()}
  elif isinstance(modules_or_tensors, (tuple, list)):
    return [obj.cuda(device_id) for obj in modules_or_tensors]
  return modules_or_tensors.cuda(device_id)
