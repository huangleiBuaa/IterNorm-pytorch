# IterNorm-pytorch
Pytorch reimplementation of the IterNorm methods, which is described in the following paper:

**Iterative Normalization: Beyond Standardization towards Efficient Whitening** 

Lei Huang, Yi Zhou, Fan Zhu, Li Liu, Ling Shao

*IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019 (accepted).*
[arXiv:1904.03441](https://arxiv.org/abs/1904.03441)


This project also provide the pytorch implementation of Decorrelated Batch Normalization (CVPR 2018, [arXiv:1804.08450](https://arxiv.org/abs/1804.08450)), more details please refer to the [Torch project](https://github.com/princeton-vl/DecorrelatedBN). 

## Requirements and Dependency
* Install [PyTorch](http://torch.ch) with CUDA (for GPU). (Experiments are validated on python 3.6.8 and pytorch-nightly 1.0.0)
* (For visualization if needed), install the dependency [visdom](https://github.com/facebookresearch/visdom) by:
```Bash
pip install visdom
 ```


## Experiments
 
 #### 1.  VGG-network on Cifar-10 datasets:
 
run the scripts in the `./cifar10/experiments/vgg`. Note that the dataset root dir should be altered by setting the para '--dataset-root', and the dataset style is described as:
```
-<dataset-root>
|-cifar10-batches-py
||-data_batch_1
||-data_batch_2
||-data_batch_3
||-data_batch_4
||-data_batch_5
||-test_batch
```
If the dataset is not exist, the script will download it, under the conditioning that the `dataset-root` dir is existed

 #### 2.  Wide-Residual-Network on Cifar-10 datasets:
 
run the scripts in the `./cifar10/experiments/wrn`. 

#### 3. ImageNet experiments.

run the scripts in the `./ImageNet/experiment`. Note that resnet18 experimetns are run on one GPU, and resnet-50/101 are run on 4 GPU in the scripts. 

Note that the dataset root dir should be altered by setting the para '--dataset-root'.
 and the dataset style is described as:
 
 ```
 -<dataset-root>
|-train
||-class1
||-...
||-class1000  
|-var
||-class1
||-...
||-class1000  
```
  
 ## Using IterNorm in other projects/tasks
  (1) copy `./extension/normalization/iterative_normalization.py` to the respective dir.
  
  (2) import the `IterNorm` class in `iterative_normalization.py`
  
  (3) generally speaking, replace the `BatchNorm` layer by `IterNorm`, or add it in any place if you want to the feature/channel decorrelated. Considering the efficiency (Note that `BatchNorm` is intergrated in `cudnn` while `IterNorm` is based on the pytorch script without optimization), we recommend 1) replace the first `BatchNorm`; 2) insert extra `IterNorm` before the first skip connection in resnet; 3) inserted before the final linear classfier as described in the paper.
  
  (4) Some tips related to the hyperparamters (Group size `G` and Iterative Number `T`). We recommend `G=64` (i.e., the channel number in per group is 64) and `T=5` by default. If you run on large batch size (e.g.>1024), you can either increase `G` or `T`. For fine tunning, fix `G=64 or G=32`, and search `T={3,4,5,6,7,8}` may help. 
