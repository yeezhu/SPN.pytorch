#!usr/bin/env bash
# select gpu devices
export CUDA_VISIBLE_DEVICES=0,1
# train
# ../data/voc/ is the path of VOCdevkit.
python -m experiment.demo_voc2007 ../data/voc/ \
--image-size 112,224,560 --batch-size 16 --lr 0.01 --epochs 20 
