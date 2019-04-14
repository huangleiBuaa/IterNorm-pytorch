#!/usr/bin/env bash
cd "$(dirname $0)/.."
python3 imagenet.py \
-a=resnet18 \
-ac=last_bn=False \
--arch-cfg=dropout=0.3 \
--batch-size=256 \
--epochs=100 \
-oo=sgd \
-oc=momentum=0.9 \
-wd=1e-4 \
--lr=0.1 \
--lr-method=step \
--lr-steps=30 \
--lr-gamma=0.1 \
--norm=ItN \
--norm-cfg=T=5,num_channels=64 \
$@