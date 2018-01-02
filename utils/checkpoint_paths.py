import os
import re
from datetime import datetime

CHKPT_EXT = 'pth'
CHKPT_REGEXP = re.compile('.+\.{}(\.[\d]+)?$'.format(CHKPT_EXT))

_FMT_RUN_DIR = '{run_name}_{time}'
_FMT_PERIODIC_CHKPT = 'periodic-chkpt_{time}_{epoch}.' + CHKPT_EXT
_FMT_BEST_CHKPT = 'best-chkpt_{time}_{epoch}_{metric:.4f}.' + CHKPT_EXT
_FMT_CONFIG = 'config_{time}.json'
_FMT_TIME = '{year}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}'


def _format_run_dir_name(**kwargs):
  log_filename = '{}_{}'.format(kwargs['run_name'],
                                kwargs['time_str'])
  return log_filename


def _format_checkpoint_name(**kwargs):
  if kwargs['epoch'] is not None:
    log_filename = '{}.{}.{}.{}'.format(kwargs['prefix'], kwargs['epoch'],
                                        kwargs['time_str'], kwargs['ext'])
  else:
    log_filename = '{}.{}.{}'.format(kwargs['prefix'], kwargs['time_str'],
                                     kwargs['ext'])
  return log_filename


def _get_path(base_dir, format_str, **kwargs):
  now = datetime.now()
  time = _FMT_TIME.format(year=now.year, month=now.month, day=now.day,
                          hour=now.hour, minute=now.minute, second=now.second)
  kwargs['time'] = time
  log_filename = format_str.format(**kwargs)
  log_base_file_path = os.path.join(base_dir, log_filename)

  # Make sure path is unique
  idx = 2
  log_file_path = log_base_file_path
  while os.path.exists(log_file_path):
    log_file_path = '{}.{}'.format(log_base_file_path, idx)
    idx += 1

  return log_file_path


def get_run_dir(base_dir, run_name):
  return _get_path(base_dir, _FMT_RUN_DIR, run_name=run_name)


def get_config_path(run_dir):
  return _get_path(run_dir, _FMT_CONFIG)


def get_periodic_checkpoint_path(run_dir, epoch):
  return _get_path(run_dir, _FMT_PERIODIC_CHKPT, epoch=epoch)


def get_best_checkpoint_path(best_dir, epoch, metric):
  return _get_path(best_dir, _FMT_BEST_CHKPT, epoch=epoch, metric=metric)


def is_checkpoint_path(path):
  return CHKPT_REGEXP.match(path) is not None
