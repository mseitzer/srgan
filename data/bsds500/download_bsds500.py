#!/usr/bin/env python
"""Download and setup the BSDS500 dataset

The images come from Arbelaez, Maire, Fowlkes & J. Malik:
Contour Detection and Hierarchical Image Segmentation (2011)
(see https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/
grouping/resources.html)
Note that the BSDS500 validation set contains the same images as in the
BSDS100 dataset.
"""
import argparse
import os
import shutil
import sys
import tarfile
import urllib.request

DATASET_URL = ('http://www.eecs.berkeley.edu/Research/Projects/CS/'
               'vision/grouping/BSR/BSR_bsds500.tgz')

parser = argparse.ArgumentParser(description='Generate BSDS500 dataset')
parser.add_argument('-f', '--force', action='store_true',
                    help='Remove existing dataset directory')
parser.add_argument('--dest-path', default='../../resources/data/BSDS500',
                    help='Path to output folder')
parser.add_argument('--from-tar', default=None,
                    help='Tarfile of the dataset to extract from')


def download_and_extract_bsds500(dest_path, from_tar_file=None):
  """Download and extract BSDS500 dataset

  Parameters
  ----------
  dest_path : string
    Destination file path
  from_tar_file : string
    If not None, use a local tarfile instead of downloading it from remote
  """
  if not os.path.exists(dest_path):
    os.makedirs(dest_path)

  tar_file = from_tar_file
  if tar_file is None:
    print('Downloading dataset from {}'.format(DATASET_URL))
    data = urllib.request.urlopen(DATASET_URL)

    tar_file = os.path.join(dest_path, os.path.basename(DATASET_URL))
    with open(tar_file, 'wb') as f:
      f.write(data.read())

  print('Extracting data')
  with tarfile.open(tar_file) as tar:
    for item in (item for item in tar if 'BSDS500/data/images' in item.name):
      item.name = os.path.relpath(item.name, 'BSR/BSDS500/data')
      tar.extract(item, dest_path)

  if from_tar_file is None:
    os.remove(tar_file)


def main(argv):
  args = parser.parse_args(argv)

  if args.force and os.path.isdir(args.dest_path):
    shutil.rmtree(args.dest_path)

  if not os.path.isdir(args.dest_path):
    download_and_extract_bsds500(args.dest_path, args.from_tar)


if __name__ == '__main__':
  main(sys.argv[1:])
