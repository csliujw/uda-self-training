# uda-self-training
This repo is the official implementation of **<a href="https://www.sciencedirect.com/science/article/pii/S1569843222001297">Weakly supervised high spatial resolution land cover mapping based on self-training with weighted pseudo-labels</a>**. The code will be release public in June 2023.

# Model zoo
The models with the scores can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1eT1Y6DFE4lqgAbrmY4bmVw) (Extraction code: 1234).

# Dataset
[LoveDA Dataset](https://github.com/Junjue-Wang/LoveDA#dataset-and-contest)

# Getting Started
## Requirments:
```shell
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

pip install mmcv-full==1.5.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

pip install mmsegmentation==0.24.1

pip install ever-beta==0.2.3

pip install timm

pip install --upgrade git+https://github.com/Z-Zheng/SimpleCV.git
```

## Prepare LoveDA Dataset
```bash
ln -s </path/to/LoveDA> ./LoveDA
```

## Train
- Stage one, training with source domain: `python source.py`
- Stage two, self-training: `python train.py`
- the oracle setting to test the upper limit of our method’s accuracy in a single domain: `python oracle.py` 

# Acknowledgments
This code is heavily borrowed from [LoveDA](https://github.com/Junjue-Wang/LoveDA).