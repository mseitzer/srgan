#!/usr/bin/env python
"""Download and setup the Set5 dataset

We are using the Set5 compilation provided by Huang, Singh & Ahuja:
Single Image Super-Resolution From Transformed Self-Exemplars (2015)
(https://github.com/jbhuang0604/SelfExSR)
The original Set5 images come from Bevilacqua, Roumy, Guillemot & Alberir:
Low-Complexity Single-Image Super-Resolution based on Nonnegative Neighbor
Embedding (2012)
"""
import argparse
import os
import shutil
import sys
import urllib.request
from zipfile import ZipFile

DATASET_URL = ('https://uofi.box.com/shared/static/'
               'kfahv87nfe8ax910l85dksyl2q212voc.zip')
FILENAME = 'Set5_SR.zip'

parser = argparse.ArgumentParser(description='Generate Set5 dataset')
parser.add_argument('-f', '--force', action='store_true',
                    help='Remove existing dataset directory')
parser.add_argument('--dest-path', default='../../resources/data/Set5',
                    help='Path to output folder')
parser.add_argument('--from-zip', default=None,
                    help='Zipfile of the dataset to extract from')


def download_and_extract_set5(dest_path, from_zip_file=None):
  """Download and extract Set5 dataset

  Parameters
  ----------
  dest_path : string
    Destination file path
  from_zip_file : string
    If not None, use a local zipfile instead of downloading it from remote
  """
  if not os.path.exists(dest_path):
    os.makedirs(dest_path)

  zip_file = from_zip_file
  if zip_file is None:
    print('Downloading dataset from {}'.format(DATASET_URL))
    data = urllib.request.urlopen(DATASET_URL)

    zip_file = os.path.join(dest_path, FILENAME)
    with open(zip_file, 'wb') as f:
      f.write(data.read())

  print('Extracting data')
  with ZipFile(zip_file) as myzip:
   for img in (img for img in myzip.infolist()
               if '_LR' in img.filename or '_HR' in img.filename):
     img.filename = os.path.relpath(img.filename, 'Set5')
     myzip.extract(img, dest_path)

  if from_zip_file is None:
    os.remove(zip_file)


def main(argv):
  args = parser.parse_args(argv)

  if args.force and os.path.isdir(args.dest_path):
    shutil.rmtree(args.dest_path)

  if not os.path.isdir(args.dest_path):
    download_and_extract_set5(args.dest_path, args.from_zip)


if __name__ == '__main__':
  main(sys.argv[1:])
