## Summary

This repository provides a reference implementation for the paper "Beyond Cartesian Representations for Local Descriptors" ([link](https://arxiv.org/abs/1908.05547)). If you use it, please cite the paper:

```
@article{Ebel19,
    andauthor = {Patrick Ebel and Anastasiia Mishchuk and Kwang Moo Yi and Pascal Fua and Eduard Trulls},
   title = {{Beyond Cartesian Representations for Local Descriptors}},
   booktitle = {Proc. of ICCV},
   year = 2019,
},
```

Please consider also citing the paper upon which the network architecture is based:

```
@article{Mishchuk17,
   author = {Anastasiya Mishchuk and Dmytro Mishkin and Filip Radenovic and Jiri Matas},
   title = {{Working hard to know your neighbor's margins: Local descriptor learning loss}},
   booktitle = {Proc. of NIPS},
   year = 2017,
}
```

## Setting up your environment

Our code relies on pytorch. Please see `system/log-polar.yml` for a list of dependencies. You can create an environment with [miniconda](https://docs.conda.io/en/latest/miniconda.html) including all the dependencies with `conda env create -f system/log-polar.yml`.

## Inference

We provide two scripts to extract descriptors given an image. Please check the notebook `example.ipynb` for an demo where you can visualize the log-polar patches. You can also run `example.py` for a command-line script that extracts features and saves them as a HDF5 file. Run `python example.py --help` for options. Keypoints are extracted with SIFT via OpenCV.

## Training

### Download the data

We rely on the data provided by the [2019 Image Matching Workshop challenge](https://image-matching-workshop.github.io/challenge/). You will need to download the following:

* Images (a): available [here](https://gfx.uvic.ca/nextcloud/index.php/s/75JNSoggacQhOkQ). We use the following sequences for training: `brandenburg_gate`, `grand_place_brussels`, `pantheon_exterior`, `taj_mahal`, `buckingham`, `notre_dame_front_facade`, `sacre_coeur`, `temple_nara_japan`, `colosseum_exterior`, `palace_of_westminster`, `st_peters_square`.
(you only need the images, feel free to delete other files).
* Ground truth match data for training (b): available [here](https://drive.switch.ch/index.php/s/UtBIgsvh05KJnch).

We generate valid matches with the calibration data and depth maps available in the IMW dataset: please refer to the paper for details. We do not provide the code to preprocess it as we are currently refactoring it.

This data should go into the `dl` folder, which contains `colmap` (a) and `patchesScale16_network_1.5px` (b).

### Train

Configuration files are stored under `configs`. You can check `init.yml` for an example. This is the default configuration file. You can specify a different one with:

```
$ python modules/hardnet/hardnet.py --config_file configs/init.yml
```

Please refer to the code for the different parameters.

## Evaluation

### AMOS patches dataset. HPatches dataset.

You can evaluate performance on the [AMOS](https://github.com/pultarmi/AMOS_patches/) and [HPatches](https://github.com/hpatches/hpatches-dataset) datasets. First, clone the dependencies with `git submodule udpate --init`, and download the weights for GeoDesc, following their instructions.
You can then run the following script, that downloads and extracts data in the appropriate format:

```
$ sh eval_amos.sh
$ sh eval_amos_other_baselines.sh

$ sh eval_hpatches.sh
$ sh eval_hpatches_other_baselines.sh
```

