#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local train on cityscapes. #
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_train_cityscapes.sh
#
#

## Use loop_train_script.py to call this script. ##
## Template version-controlled script. Copy this script and modify. ##

# Exit immediately if a command exits with a non-zero status.
set -e

export CUDA_VISIBLE_DEVICES="0"

#################################################################################################
## If using S3, make sure ~/.aws/credentials contains aws_access_key_id and aws_secret_access_key
## Make sure region is set correctly
export AWS_REGION="us-east-2"
AWS_BUCKET=""  # set to "" if not using s3

LR=0.0005
EXP_FOLDER="exp/train_fullres_dlv3-clh_dynamic_B10_every_other_block_maskvoid-100kit-LR${LR}"
MODEL_VARIANT="xception_65"

DATASET_FOLDER="cityscapes"
#################################################################################################

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

if [[ ${AWS_BUCKET} = "s3:"* ]]; then
	BACKUP_DIR=${AWS_BUCKET}
else
	BACKUP_DIR=${WORK_DIR}
fi
# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
DATASET_DIR="datasets"

TFRECORD_FOLDER="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/tfrecord"

TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/train"
# Save checkpoints to AWS if enabled. Else save to WORK_DIR
TRAIN_CHKPT_BACKUP_LOGDIR="${BACKUP_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/train/backup_checkpoints"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/export"

INIT_FOLDER="${WORK_DIR}/pretrained_models"
#################################################################################################
#INIT_CHECKPOINT="${INIT_FOLDER}/model.ckpt-90000"
INIT_CHECKPOINT="${INIT_FOLDER}/deeplabv3_pascal_train_aug_22chgaussinit/model.ckpt"
#################################################################################################

if [[ ${AWS_BUCKET} = "s3:"* ]]; then
	mkdir -p "${INIT_FOLDER}"
	mkdir -p "${TRAIN_LOGDIR}"
	mkdir -p "${EVAL_LOGDIR}"
	mkdir -p "${VIS_LOGDIR}"
	mkdir -p "${EXPORT_DIR}"
else
	mkdir -p "${TRAIN_CHKPT_BACKUP_LOGDIR}"
	mkdir -p "${INIT_FOLDER}"
	mkdir -p "${TRAIN_LOGDIR}"
	mkdir -p "${EVAL_LOGDIR}"
	mkdir -p "${VIS_LOGDIR}"
	mkdir -p "${EXPORT_DIR}"
fi

cd "${CURRENT_DIR}"

NUM_ITERATIONS=$1
EVAL_INTERVAL=$2
STOP_ITERATION=$3  # Stop iteration should be max(num_iterations, next_eval_iteration). This will be used to copy backup checkpoints.

# This trains block annotation model. Hints are provided to model by sampling the ground truth label given. 
# To replicate experiment in paper, ground truth label should be a 50% block annotation (i.e., 50% of blocks are labelled per image)
python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train_block" \
  --model_variant=${MODEL_VARIANT} \
  --base_learning_rate=${LR} \
  --train_crop_size=769 \
  --train_crop_size=769 \
  --train_batch_size=2 \
  --batch_iter=1 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --validation_interval="${EVAL_INTERVAL}" \
  --fine_tune_batch_norm=False \
  --initialize_last_layer=False \
  --last_layers_contain_logits_only=True \
  --tf_initial_checkpoint="${INIT_CHECKPOINT}" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${TFRECORD_FOLDER}" \
  --dataset=cityscapes \
  --input_hints=True \
  --hint_types=dynamic_block_hint \
  --dynamic_block_hint_B=10 \
  --dynamic_block_hint_p=0.5 \
  --class_balanced_loss=False \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --resize_factor=16 \
  --decoder_output_stride=4 \


if [[ ${AWS_BUCKET} = "s3:"* ]]; then
	ls ${TRAIN_LOGDIR}/model.ckpt-${STOP_ITERATION}* | xargs -Ifile aws s3 cp file ${TRAIN_CHKPT_BACKUP_LOGDIR}/
else
	cp -v ${TRAIN_LOGDIR}/model.ckpt-${STOP_ITERATION}* ${TRAIN_CHKPT_BACKUP_LOGDIR}
fi


##### EVALUATION NOTE #####
# To eval, run on val split
# - ensure ground truth labels in val tfrecord are 50% block annotations
# - set dynamic_block_hint_p=1.0 (so all ground truth labels are given as a hint)
# - compue mIOU against full val ground truth labels (for convenience, can use compute_miou.py)
################

VIS_LOGDIR="${VIS_LOGDIR}/${STOP_ITERATION}"
mkdir -p "${VIS_LOGDIR}"

# Produce full-image labels for training set.
# Set vis_num_batches=2975 to visualize for all training images.
python "${WORK_DIR}"/vis.py \
  --logtostderr \
  --vis_split="train_block" \
  --model_variant=${MODEL_VARIANT} \
  --vis_crop_size=1025 \
  --vis_crop_size=2049 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --vis_logdir="${VIS_LOGDIR}" \
  --dataset_dir="${TFRECORD_FOLDER}" \
  --dataset=cityscapes \
  --max_number_of_iterations=1 \
  --colormap_type=cityscapes \
  --input_hints=True \
  --hint_types=dynamic_block_hint \
  --dynamic_block_hint_B=10 \
  --dynamic_block_hint_p=1.0 \
  --vis_num_batches=25 \
  --atrous_rates=12 \
  --atrous_rates=24 \
  --atrous_rates=36 \
  --output_stride=8 \
  --resize_factor=8 \
  --decoder_output_stride=4 \

if [[ ${AWS_BUCKET} = "s3:"* ]]; then
	aws s3 sync ${VIS_LOGDIR} ${TRAIN_CHKPT_BACKUP_LOGDIR}/${STOP_ITERATION}
fi


