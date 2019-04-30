#!/usr/bin/env bash
cd "$(dirname $0)/../.."
CUDA_VISIBLE_DEVICES=0 python3 cifar10.py \
-a=vgg \
--batch-size=256 \
--epochs=160 \
-oo=sgd \
-oc=momentum=0.9 \
-wd=0 \
--lr=1 \
--lr-method=steps \
--lr-steps=60,120 \
--lr-gamma=0.2 \
--dataset-root=/home/lei/PycharmProjects/data/cifar10/ \
--norm=ItN \
--norm-cfg=T=5,num_channels=512 \
--seed=1 \
--log-suffix=LargeLR \
$@
