# Installation

## Install PyTorch in Conda env

```shell
# create conda env
conda create -n detectron2 python=3.6

# activate the enviorment
conda activate detectron2

# install PyTorch v1.4 with GPU
conda install pytorch torchvision -c pytorch
# or without GPU (same command)
# conda install pytorch torchvision -c pytorch

```

## Build Detectron2 from Source

```shell
# install it from a local clone:
git clone git@github.com:chihyaoma/detectron2-semi.git
python -m pip install -e detectron2-semi

# Or if you are on macOS
git clone git@github.com:chihyaoma/detectron2-semi.git
CC=clang CXX=clang++ python -m pip install -e detectron2-semi
```

## Download COCO dataset

```shell
# download images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

## Install TensorBoard in Conda env (optional)

```shell
# create conda env
conda create -n tb python=3.6

# activate the enviorment
conda activate tb

# install TensorBoard
pip install tensorboard
```
