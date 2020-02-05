#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

export CUDA_VISIBLE_DEVICES=""

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

## Run model_test first to make sure the PYTHONPATH is correctly set.
#python "${WORK_DIR}"/model_test.py -v
#exit

DATASET_DIR="datasets"

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
DATASET_FOLDER="cityscapes"
#MODEL_VARIANT="xception_65"
#EXP_FOLDER="exp/default_train_finetune"
MODEL_VARIANT="mobilenet_v2"
#EXP_FOLDER="exp/default_train_finetune_mobilenet"
EXP_FOLDER="exp/train_finetune_mobilenet_scale16_batchnorm"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/init_models"
#INIT_CHECKPOINT="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt"
INIT_CHECKPOINT="${INIT_FOLDER}/deeplabv3_mnv2_pascal_trainval/model.ckpt-30000"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/train"
TRAIN_CHKPT_BACKUP_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/train/backup_checkpoints"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${TRAIN_CHKPT_BACKUP_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

## Copy locally the trained checkpoint as the initial checkpoint.
#TF_INIT_ROOT="http://download.tensorflow.org/models"
#TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
#cd "${INIT_FOLDER}"
#wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
#tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

TFRECORD_FOLDER="${WORK_DIR}/${DATASET_DIR}/${DATASET_FOLDER}/tfrecord"

#NUM_ITERATIONS=10000
#NUM_ITERATIONS=$@

python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val_fullres" \
  --model_variant=${MODEL_VARIANT} \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --resize_factor=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=1025 \
  --eval_crop_size=2049 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${TFRECORD_FOLDER}" \
  --dataset=cityscapes \
  --max_number_of_evaluations=-1 \
  --eval_interval_secs=900 \ 
