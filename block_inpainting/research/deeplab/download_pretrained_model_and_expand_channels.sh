#!/bin/bash

mkdir pretrained_models
cd pretrained_models
wget http://download.tensorflow.org/models/deeplabv3_pascal_train_aug_2018_01_04.tar.gz
tar xzvf deeplabv3_pascal_train_aug_2018_01_04.tar.gz
cd ../

bash ./weight_transfer_cityscapes.sh
