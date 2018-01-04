#!/bin/sh

mkdir -p resources/data
mkdir -p resources/images
mkdir -p resources/models

echo "Setting up BSDS500 dataset"
data/bsds500/download_bsds500.py --dest-path=resources/data/BSDS500

echo "Setting up Set5 dataset"
data/set5/download_set5.py --dest-path=resources/data/Set5

echo "Setting up Set14 dataset"
data/set14/download_set14.py --dest-path=resources/data/Set14

