#!usr/bin/env bash
# select gpu devices
export CUDA_VISIBLE_DEVICES=0,1
# train
python3 -m experiment.demo_voc2007 ../data/voc/ \
--image-size 224 --batch-size 64 --lr 0.01 --epochs 20 
