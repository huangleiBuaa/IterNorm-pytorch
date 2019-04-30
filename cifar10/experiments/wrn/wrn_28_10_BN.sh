#!/usr/bin/env bash
cd "$(dirname $0)/../.."
CUDA_VISIBLE_DEVICES=0 python3 cifar10.py \
-a=WRN_28_10 \
--arch-cfg=dropout=0.3 \
--batch-size=128 \
--epochs=200 \
-oo=sgd \
-oc=momentum=0.9 \
-wd=5e-4 \
--lr=0.1 \
--lr-method=steps \
--lr-steps=60,120,160 \
--lr-gamma=0.2 \
--dataset-root=/home/lei/PycharmProjects/data/cifar10/ \
--norm=BN \
--norm-cfg=T=5,num_channels=64 \
--seed=1 \
--log-suffix=seed1 \
$@
