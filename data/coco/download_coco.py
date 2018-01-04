#!/usr/bin/env python
"""Download and setup the MS COCO dataset

We are using the 2017 version of the dataset (see http://cocodataset.org)
- train: 118287 images
- val: 5000 images
- test: 40670 images
"""
import argparse
import os
import shutil
import sys
import urllib.request
from zipfile import ZipFile

DATASET_URLS = {
    'train': 'http://images.cocodataset.org/zips/train2017.zip',
    'val': 'http://images.cocodataset.org/zips/val2017.zip',
    'test': 'http://images.cocodataset.org/zips/test2017.zip'
}

parser = argparse.ArgumentParser(description='Generate COCO dataset')
parser.add_argument('-f', '--force', action='store_true',
                    help='Remove existing dataset directory')
parser.add_argument('--dest-path', default='../../resources/data/COCO',
                    help='Path to output folder')
parser.add_argument('--from-zip', default=None,
                    help='Zipfile of the dataset to extract from')
parser.add_argument('fold', default='train', choices=['train', 'val', 'test'],
                    help='Dataset fold to download')

def download_and_extract_coco(dest_path, fold, from_zip_file=None):
  """Download and extract COCO dataset

  Parameters
  ----------
  dest_path : string
    Destination file path
  fold : string
    Dataset fold to use
  from_zip_file : string
    If not None, use a local zipfile instead of downloading it from remote
  """
  dest_path = os.path.join(dest_path, 'images')
  if not os.path.exists(dest_path):
    os.makedirs(dest_path)

  zip_file = from_zip_file
  if zip_file is None:
    url = DATASET_URLS[fold]
    print('Downloading dataset {} fold from {}'.format(fold, url))
    data = urllib.request.urlopen(url)

    zip_file = os.path.join(dest_path, FILENAME)
    with open(zip_file, 'wb') as f:
      f.write(data.read())

  print('Extracting data')
  with ZipFile(zip_file) as myzip:
   for img in (img for img in myzip.infolist() if '.jpg' in img.filename):
     myzip.extract(img, dest_path)

  if from_zip_file is None:
    os.remove(zip_file)


def main(argv):
  args = parser.parse_args(argv)

  if args.force and os.path.isdir(args.dest_path):
    shutil.rmtree(args.dest_path)

  if not os.path.isdir(args.dest_path):
    download_and_extract_coco(args.dest_path, args.fold, args.from_zip)


if __name__ == '__main__':
  main(sys.argv[1:])
