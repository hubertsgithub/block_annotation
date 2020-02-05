### INFO ###
Template code is provided here to train block inpainting model on Cityscapes. Code is based on official Deeplabv3+ Tensorflow implementation.  This codebase may contain traces of deprecated code. If you find any issues, please let me know or submit a pull request.

This version of code runs with: Python 3.7, Ubuntu 16, Tensorflow 1.14, CUDA 10.2, TITAN Xp. Have not tested model performance.
Original results from paper: Python 2.7, Ubuntu 14, Tensorflow 1.8, Cuda 9, TITAN Xp. Model performance reported in paper.

If our work is useful to you, please consider citing our paper.

Note that block inpainting should be deployed under the following conditions:
1. Data is partially annotated with block annotations. In our paper, we show results with 50% block annotations.
2. Block inpainting network is trained on the annotations in (1). Refer to paper for details.
3. Inference is performed with the annotations in (1) as input to block inpainting netowrk. This produces full image labels.

The block inpainting network must be trained on the dataset that is being annotated. The network is not intended to generalize to new datasets or new classes. It is not a class agnostic method as it assigns class labels to every pixel.


### INSTRUCTIONS: ###
1. Download Cityscapes and convert labels to train IDs (see https://github.com/mcordts/cityscapesScripts, use the CreateTrainIdLabelImgs script.)
2. Create block annotations from Cityscapes labels (to synthetically create dataset with block annotated labels).
	- cd research/deeplab/datasets
	- python generate_every_other_blocks_fixed_size.py
3. Create tfrecord.
	- cd research/deeplab/datasets
	- python build_cityscapes_data.py
4. Convert pretrained weights to accomodate additional channels.
	- Provided script will download pretrained deeplab model weights on pascal voc train aug.
	- cd research/deeplab
	- bash download_pretrained_model_and_expand_channels.sh
5. Train network.
	- cd research/deeplab
	- python loop_train_script.py local_train_cityscapes_dlv3_block_annotation.sh --num_iterations 100000 --eval_interval 10000 --start_iteration 0
6. See training script for visualization code to inpaint full image labels from block annotated images with trained network. Please refer to contents of local_train_cityscapes_dlv3_block_annotation.sh for more details.






