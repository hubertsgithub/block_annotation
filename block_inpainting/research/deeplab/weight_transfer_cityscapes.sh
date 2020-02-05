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

echo "#####################################"
echo "THE PURPOSE OF THIS SCRIPT IS TO  TRANSFER WEIGHTS FROM A CHECKPOINT."
echo "#####################################"

# Exit immediately if a command exits with a non-zero status.
set -e

export CUDA_VISIBLE_DEVICES="0"

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Set up the working directories.
MODEL_FOLDER="${WORK_DIR}/pretrained_models"

##################################################################################
MODEL_VARIANT="xception_65"

SOURCE_CHECKPOINT_NAME="model.ckpt"
SOURCE_CHECKPOINT_DIR="${MODEL_FOLDER}/deeplabv3_pascal_train_aug"
OUTPUT_CHECKPOINT_DIR="${MODEL_FOLDER}/deeplabv3_pascal_train_aug_22chgaussinit"

# Set dataset to dataset of source checkpoint.
NUM_CLASSES=21              # Num classes in pretrained checkpoint... required to initialize model graph.
INPUT_CHANNELS="22"  		# Max(3, INPUT_CHANNELS)
INPUT_KERNEL_FILLER="gaussian"	# zeros, gaussian
##################################################################################

cd "${CURRENT_DIR}"

python "${WORK_DIR}"/weight_transfer_deeplab.py \
  --logtostderr \
  --model_variant="${MODEL_VARIANT}" \
  --train_crop_size=128 \
  --train_crop_size=192 \
  --source_checkpoint_name="${SOURCE_CHECKPOINT_NAME}" \
  --source_checkpoint_dir="${SOURCE_CHECKPOINT_DIR}" \
  --output_checkpoint_dir="${OUTPUT_CHECKPOINT_DIR}" \
  --num_classes="${NUM_CLASSES}"\
  --input_channels="${INPUT_CHANNELS}" \
  --input_kernel_filler="$INPUT_KERNEL_FILLER" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --resize_factor=16 \
  --decoder_output_stride=4 \

