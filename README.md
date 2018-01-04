# SRGAN and SRResNet: Super-Resolution using GANs

This is a complete Pytorch implementation of [Christian Ledig et al: "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802).

So far, a SRResNet pretrained on the COCO dataset (118k images) is provided. A pretrained SRGAN will follow shortly.

## Setup and Usage

### Prerequisites
- Get all dependencies using [Conda](https://conda.io): `conda env create -n srgan -f environment.yml`
- Activate the conda environment: `source activate srgan`
- Setup the folder structure and download the evaluation datasets: `./setup.sh`
- If you want to train a model yourself, you can download the MS COCO dataset: 
```
data/coco/download_coco.py --dest-path=resources/data/COCO
```
Warning: the dataset is 18GB in size and will take some time to download!

### Super-resolving an image using provided pretrained checkpoints

```
./eval.py -i configs/srresnet.json resources/pretrained/srresnet.pth path/to/image.jpg
```
The super-resolved image will be saved to the same folder as the input image and named `image_pred.jpg`.

Note that the script uses GPU 0 by default. To use the CPU, pass `-c ''` as an additional flag to the script.

### Evaluating the provided pretrained checkpoints

To reproduce the [score evaluations](#quantitative-results) of the benchmark datasets:
```
./eval.py configs/srresnet.json resources/pretrained/srresnet.pth Set5 Set14 BSDS500
```
To also get the super-resolved images of the benchmark dataset, you can pass the infer flag `-i` to the script.

### Training

The following commands reproduce the pretrained checkpoints.
```
./train.py configs/srresnet.json
```
Note that you need to download the COCO train set beforehand.

Alternatively, you can train on the 200 training images of the BSDS500 dataset:
```
./train.py --conf train_dataset=BSDS500 configs/srresnet.json
```

Some configuration values you can tweak:
- `upscale_factor`: Upscaling factor the network is trained on
- `num_epochs`: Number of epochs the networks is trained

## Results

All given results are taken at 4x scale.

### Quantitative results

PSNR and SSIM scores of this implementation compared against the values reported in the paper. 
This implementation reaches slightly lower scores than the reference, which might be because of the 
larger training set Ledig et al used (350k ImageNet images vs. 118k COCO images).

| Dataset | Bicubic        | SRResnet (Ledig et al) | SRResNet (ours) |
| ------- | -------------- | ---------------------- | --------------- |
| Set5    | 28.43 / 0.8211 | 32.05 / 0.9019         | 31.94 / 0.8959  |
| Set14   | 25.99 / 0.7486 | 28.49 / 0.8184         | 27.75 / 0.7690  |
| BSDS100 | 25.94 / 0.6935 | 27.58 / 0.7620         | 27.55 / 0.7445  |
