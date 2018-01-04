#!/usr/bin/env python
import argparse
import os
import sys

from torch.utils.data import DataLoader
from torchvision.utils import save_image

import utils
from data import load_dataset, is_dataset
from data.datasets import make_sr_dataset_from_folder
from data.transform_wrappers import get_sr_transform
from training import build_runner
from utils.checkpoints import restore_checkpoint
from utils.checkpoint_paths import get_run_dir
from utils.config import Configuration

DEFAULT_NUM_WORKERS = 2

parser = argparse.ArgumentParser(description=('Validate model and infer'
                                              ' predictions on images'))
parser.add_argument('-c', '--cuda', default='0', type=str, help='GPU to use')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Print more info')
parser.add_argument('--data-dir', default='resources/data',
                    help='Path to data directory')
parser.add_argument('--out-dir', default='resources/images',
                    help='Path to where to save outputs')
parser.add_argument('-i', '--infer', action='store_true',
                    help='Save predicted images')
parser.add_argument('-d', '--dump', action='store_true',
                    help='Save input, target and predicted images')
parser.add_argument('-f', '--fold', choices=['train', 'val', 'test'],
                    default='val', help='Fold of dataset to use')
parser.add_argument('--conf', nargs='+',
                    help=('Optional config values to set'
                          'The format is "key=value"'))
parser.add_argument('config', help='Config file to use')
parser.add_argument('checkpoint', help='Checkpoint to use weights from')
parser.add_argument('files_or_dirs', nargs='*',
                    help='Files or folders to evaluate')


def main(argv):
  args = parser.parse_args(argv)

  if args.cuda != '':
    try:
      args.cuda = utils.set_cuda_env(args.cuda)
    except Exception:
      print('No free GPU on this machine. Aborting run.')
      return
    print('Running on GPU {}'.format(args.cuda))

  # Load configuration
  conf = Configuration.from_json(args.config)
  conf.args = args
  if args.conf:
    new_conf_entries = {}
    for arg in args.conf:
      key, value = arg.split('=')
      new_conf_entries[key] = value
    conf.update(new_conf_entries)
  if args.verbose:
    print(conf)

  utils.set_random_seeds(conf.seed)

  # Setup model
  runner = build_runner(conf, conf.runner_type, args.cuda, mode='test')

  # Handle resuming from checkpoint
  if args.checkpoint != 'NONE':
    if os.path.exists(args.checkpoint):
      _ = restore_checkpoint(args.checkpoint, runner, cuda=args.cuda)
      print('Restored checkpoint from {}'.format(args.checkpoint))
    else:
      print('Checkpoint {} to restore from not found'.format(args.checkpoint))
      return

  # Evaluate on full image, not crops
  conf.full_image = True

  # Load datasets
  mode = 'dataset'
  if len(args.files_or_dirs) == 0:
    datasets = [load_dataset(conf, args.data_dir, conf.validation_dataset, args.fold)]
  else:
    datasets = []
    for f in args.files_or_dirs:
      if is_dataset(f):
        dataset = load_dataset(conf, args.data_dir, f, args.fold)
        datasets.append(dataset)
      else:
        mode = 'image'
        transform = get_sr_transform(conf, 'test', downscale=False)
        datasets = [make_sr_dataset_from_folder(conf, f, transform,
                                                inference=True)
                    for f in args.files_or_dirs]

  num_workers = conf.get_attr('num_data_workers', default=DEFAULT_NUM_WORKERS)

  # Evaluate all datasets
  for dataset in datasets:
    loader = DataLoader(dataset=dataset,
                        num_workers=num_workers,
                        batch_size=1,
                        shuffle=False)

    if mode == 'dataset':
      data, _, val_metrics = runner.validate(loader, len(loader))

      print('Average metrics for {}'.format(dataset.name))
      for metric_name, metric in val_metrics.items():
        print('     {}: {}'.format(metric_name, metric))
    else:
      data = runner.infer(loader)

    if args.infer or args.dump:
      if mode == 'dataset':
        output_dir = get_run_dir(args.out_dir, dataset.name)
        if not os.path.isdir(output_dir):
          os.mkdir(output_dir)

      file_idx = 0
      for batch in data:
        if mode == 'image':
          output_dir = os.path.dirname(dataset.images[file_idx])

        named_batch = runner.get_named_outputs(batch)
        inputs = named_batch['input']
        predictions = named_batch['prediction']
        targets = named_batch['target']
        for (inp, target, prediction) in zip(inputs, targets, predictions):
          image_file = os.path.basename(dataset.images[file_idx])
          name, _ = os.path.splitext(image_file)
          file_idx += 1

          if args.dump:
            input_file = os.path.join(output_dir,
                                      '{}_input.png'.format(name))
            save_image(inp.data, input_file)
            target_file = os.path.join(output_dir,
                                       '{}_target.png'.format(name))
            save_image(target.data, target_file)
          pred_file = os.path.join(output_dir,
                                   '{}_pred.png'.format(name))
          save_image(prediction.data, pred_file)


if __name__ == '__main__':
  main(sys.argv[1:])
