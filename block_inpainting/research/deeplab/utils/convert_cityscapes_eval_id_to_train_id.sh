#!/bin/bash

python convert_cityscapes_eval_id_to_train_id.py \
	--input_dir ../datasets/cityscapes/exp/train_fullres_mnv2_imbal_100-bdythr_3-clh_300_bdythr_3-100kit-LR0.01/vis/100000/train/raw_segmentation_results/ \
	--output_dir ../datasets/cityscapes/exp/train_fullres_mnv2_imbal_100-bdythr_3-clh_300_bdythr_3-100kit-LR0.01/vis/100000/train/raw_segmentation_results_train_id


python convert_cityscapes_eval_id_to_train_id.py \
	--input_dir ../datasets/cityscapes/exp/train_fullres_mnv2_imbal_500-bdythr_3-clh_300_bdythr_3-100kit-LR0.01/vis/100001/train/raw_segmentation_results/ \
	--output_dir ../datasets/cityscapes/exp/train_fullres_mnv2_imbal_500-bdythr_3-clh_300_bdythr_3-100kit-LR0.01/vis/100001/train/raw_segmentation_results_train_id
