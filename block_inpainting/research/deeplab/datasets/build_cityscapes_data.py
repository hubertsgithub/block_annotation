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
# limitations under the License.  ==============================================================================

"""Converts Cityscapes data to TFRecord file format with Example protos.

The Cityscapes dataset is expected to have the following directory structure:

  - build_cityscapes_data.py (current working directiory).
  - build_data.py
  + cityscapes
     + leftImg8bit
       + train
       + val
       + test
     + gtFine
       + train
       + val
       + test
     + tfrecord

This script converts data into sharded data files and save at tfrecord folder.

Note that before running this script, the users should (1) register the
Cityscapes dataset website at https://www.cityscapes-dataset.com to
download the dataset, and (2) run the script provided by Cityscapes
`preparation/createTrainIdLabelImgs.py` to generate the training groundtruth.

Also note that the tensorflow model will be trained with `TrainId' instead
of `EvalId' used on the evaluation server. Thus, the users need to convert
the predicted labels to `EvalId` for evaluation on the server. See the
vis.py for more details.

The Example proto contains the following fields:

  image/encoded: encoded image content.
  image/filename: image filename.
  image/format: image file format.
  image/height: image height.
  image/width: image width.
  image/channels: image channels.
  image/segmentation/class/encoded: encoded semantic segmentation content.
  image/segmentation/class/format: semantic segmentation file format.
"""
import glob
import fnmatch
import math
import os.path
#import re
import sys
import build_data
import tensorflow as tf
import numpy as np
import PIL.Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('cityscapes_root',
                           './cityscapes',
                           'Cityscapes dataset root folder.')

tf.app.flags.DEFINE_string(
    'output_dir',
    './cityscapes/tfrecord',
    'Path to save converted tfrecord of TensorFlow examples.')


# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'leftImg8bit',
    'label': 'gtFine',
    'label_coarse': 'gtCoarse'

}

# A map from data type to filename postfix.
_POSTFIX_MAP = {
    'image': '_leftImg8bit',
    'label': '_gtFine_labelTrainIds',
    'label_coarse': 'gtCoarse_labelTrainIds',

}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
    'label_coarse': 'png',
}

def _get_files(data, dataset_split):
    """Gets files for the specified data type and dataset split.

    Args:
      data: String, desired data ('image' or 'label').
      dataset_split: String, dataset split ('train', 'val', 'test')

    Returns:
      A list of sorted file names or None when getting label for
        test set.
    """
    if data == 'label' and dataset_split == 'test':
        return None

    pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
    # cityscapes/imgs_preproc/datasplit/subdir/imgs.png
    search_files = os.path.join(
        FLAGS.cityscapes_root, _FOLDERS_MAP[data], dataset_split, '*', pattern)

    filenames = glob.glob(search_files)
    if len(filenames) <= 0:
        raise Exception("NO FILES FOUND AT: {}".format(search_files))

    return sorted(filenames)


def _convert_dataset(dataset_split, dataset_name='',
                     class_ignore_value=255,
                     num_shards=1,
                     num_imgs=-1,
                     remaining_imgs_type=None,
                     remaining_imgs_num_or_ratio=None,
                     overwrite=False,
                     verbose=True,
                     shuffle=False,
                     shuffle_seed=1234):
    """Converts the specified dataset split to TFRecord format.

    Args:
      dataset_split: The dataset split (e.g., train, val).
      dataset_name: The dataset name (e.g., train_hints, val_hints). Default is set to dataset_split. This is the name of the tfrecord.

      remaining_imgs_type: if num_imgs is set, what should we do we remaining images if remaining_imgs_num_or_ratio is set?
        None: Use remaining image labels as-is
        path -- string: Use labels from path (use alternative labels)

      remaining_imgs_num_or_ratio::
        None: don't use remaining images (same as 0 or 0.0)
        ratio -- float between 0 and 1: use ratio * num_remaining_imgs remaining images.
        num -- integer: use num remaining images

    Raises:
      RuntimeError: If loaded image and label have different shape, or if the
        image file with specified postfix could not be found.
    """
    sys.stdout.flush()
    print('###############################################')
    sys.stdout.write('\rWorking on: {}\n'.format(dataset_name))



    image_files = []
    label_files = []

    print ('Using full res images and labels...')
    image_files.extend(_get_files('image', dataset_split))
    if 'coarse' in dataset_name:
        label_files.extend(_get_files('label_coarse', dataset_split))
    else:
        label_files.extend(_get_files('label', dataset_split))

    if num_imgs < 0:
        num_images = len(image_files)
    else:
        num_images = num_imgs

    remaining_imgs_label_files = None
    num_remaining_imgs_num = None
    if remaining_imgs_num_or_ratio is not None:

        if type(remaining_imgs_num_or_ratio) == float:
            assert 0 <= remaining_imgs_num_or_ratio <= 1
            num_remaining_images = int(remaining_imgs_num_or_ratio * (len(image_files) - num_images))
        if type(remaining_imgs_num_or_ratio) == int:
            assert 0 <= remaining_imgs_num_or_ratio
            num_remaining_images = min(remaining_imgs_num_or_ratio,  (len(image_files) - num_images))

        if remaining_imgs_type is None:
            remaining_imgs_label_files = list(label_files)
        elif type(remaining_imgs_type) == str:
            print ("Searching {} for label files.")
            remaining_imgs_label_files = []
            for root, dirnames, filenames in os.walk(remaining_imgs_type):
                for filename in fnmatch.filter(filenames, "*"):
                    remaining_imgs_label_files.append(os.path.join(root, filename))

            remaining_imgs_label_files = sorted(remaining_imgs_label_files)
            assert len(remaining_imgs_label_files) == len(label_files), 'Expected {} alternative label files; found {}'.format(len(label_files), len(remaining_imgs_label_files))
        else:
            raise TypeError("remaining_imgs_type should be a string or None")


    if shuffle:
        shuffled_idxs = np.arange(len(image_files))
        np.random.seed(shuffle_seed)
        np.random.shuffle(shuffled_idxs)

        print ('Using indices {} ...'.format(shuffled_idxs[:10]))

        # Image, label, boundary distance map
        image_files = np.array(image_files)[shuffled_idxs]
        label_files = np.array(label_files)[shuffled_idxs]

        # Alternative label files
        if remaining_imgs_label_files is not None:
            remaining_imgs_label_files = np.array(remaining_imgs_label_files)[shuffled_idxs]

            # Concat num_images label_files with num_remaining_images_label_files
            label_files = list(label_files)[:num_images] + list(remaining_imgs_label_files)[num_images:num_images+num_remaining_images]

            assert len(label_files) == num_images + num_remaining_images, (len(label_files), num_images, num_remaining_images)
            num_images = num_images + num_remaining_images

    if not shuffle and remaining_imgs_label_files is not None:
        raise NotImplementedError("This is not going to work; check the code")

    num_per_shard = int(math.ceil(num_images / float(num_shards)))

    image_reader = build_data.ImageReader('png', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)

    for shard_id in range(num_shards):
        if dataset_name == '':
            dataset_name = dataset_split

        shard_filename = '%s-%05d-of-%05d.tfrecord' % (
            dataset_name, shard_id, num_shards)
        output_filename = os.path.join(FLAGS.output_dir, shard_filename)
        if os.path.exists(output_filename) and not overwrite:
            print ('File exists. Skipping. {}'.format(output_filename))
            continue

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)

            for i in range(start_idx, end_idx):
                sys.stdout.flush()
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_images, shard_id))
                #sys.stdout.flush()

                # Read the image.
                image_data = tf.gfile.FastGFile(image_files[i], 'rb').read()
                height, width = image_reader.read_image_dims(image_data)

                # Read the semantic segmentation annotation.
                seg_data = tf.gfile.FastGFile(label_files[i], 'rb').read()
                seg_height, seg_width = label_reader.read_image_dims(seg_data)
                if verbose:
                    sys.stdout.write('\r\nUsing\n {}\n {}\n'.format(image_files[i], label_files[i]))

                if height != seg_height or width != seg_width:
                    raise RuntimeError(
                        'Shape mismatched between image and label.')

                # Convert to tf example.
                base_filename = os.path.basename(image_files[i]).\
                                    replace(_POSTFIX_MAP['image']+'.'+_DATA_FORMAT_MAP['image'], '')
                filename = base_filename

                example = build_data.image_seg_to_tfexample(
                    image_data=image_data,
                    filename=filename,
                    height=height,
                    width=width,
                    seg_data=seg_data,
                    )
                tfrecord_writer.write(example.SerializeToString())

        sys.stdout.write('\n')
        sys.stdout.flush()


def main(unused_argv):
    # Make sure these directories exist. tfrecord can be empty.
    assert os.path.exists(os.path.join('cityscapes', 'tfrecord')), 'Please create ./cityscapes/tfrecord directory'
    assert os.path.exists(os.path.join('cityscapes', 'leftImg8bit')), 'Please symlink leftImg8bit to ./cityscapes/leftImg8bit'
    assert os.path.exists(os.path.join('cityscapes', 'gtFine')), 'Please symlink gtFine to ./cityscapes/gtFine'

    _NUM_SHARDS = 1
    # Only support converting 'train' and 'val' sets for now.
    for dataset_split in ['train', 'val']:
        _convert_dataset(dataset_split=dataset_split,
                         num_shards=_NUM_SHARDS,
                         shuffle=True,
                         shuffle_seed=1234,
                         overwrite=True)

# To evaluate on validation set:
# -  use run network with block-annotated val labels
# -  compare predictions with fully annotated val labels






if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
    tf.app.run()
