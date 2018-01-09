#!/usr/bin/env python
"""Script to prepare a checkpoint for deployment

Strips unnecessary checkpoint data only needed to continue training,
such as the optimizer state.
"""
import argparse
import sys

import torch

from utils.checkpoints import inference_checkpoint_from_training_checkpoint

parser = argparse.ArgumentParser(description=('Prepare checkpoint for '
                                              'deployment'))
parser.add_argument('--runner_type', default='standard',
                    choices=['standard', 'adversarial'],
                    help='Runner type used to train the checkpoint')
parser.add_argument('in_checkpoint', help='Path to checkpoint file')
parser.add_argument('out_checkpoint', help='Path to output checkpoint file')


def main(argv):
  args = parser.parse_args(argv)

  # Always restore weights on CPU
  # They will also initially reside on CPU in the inference checkpoint
  map_location = lambda storage, loc: storage
  chkpt = torch.load(args.in_checkpoint, map_location=map_location)

  out_chkpt = inference_checkpoint_from_training_checkpoint(chkpt,
                                                            args.runner_type)
  torch.save(out_chkpt, args.out_checkpoint)


if __name__ == '__main__':
  main(sys.argv[1:])
