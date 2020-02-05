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
"""Evaluation script for the DeepLab model.

See model.py for more details and usage.
"""

import math
import six
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import train_utils

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('eval_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for evaluating the model.

flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('eval_crop_size', [513, 513],
                           'Image crop size [height, width] for evaluation.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during
# training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

# Dataset settings.

flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')



def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Get dataset-dependent information.
    dataset = segmentation_dataset.get_dataset(
        FLAGS.dataset, FLAGS.eval_split,
        dataset_dir=FLAGS.dataset_dir,
        use_input_hints=FLAGS.input_hints,
        hint_types=FLAGS.hint_types)

    tf.gfile.MakeDirs(FLAGS.eval_logdir)
    tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

    if FLAGS.force_dropout:
        raise Exception("Need to implement force dropout for eval.py")

    with tf.Graph().as_default():
        samples = input_generator.get(
            dataset,
            FLAGS.eval_crop_size,
            FLAGS.eval_batch_size,
            min_resize_value=FLAGS.min_resize_value,
            max_resize_value=FLAGS.max_resize_value,
            resize_factor=FLAGS.resize_factor,
            dataset_split=FLAGS.eval_split,
            is_training=False,
            model_variant=FLAGS.model_variant)

        if FLAGS.input_hints:
            ###
            # TODO: Can modify this to checkerboard block hints.
            if 'dynamic_class_partial_boundary_hint' in FLAGS.hint_types:
                assert len(FLAGS.hint_types) == 1, 'When using dynamic partial boundary class hints, do not use other hint types!'
                print("----")
                print("eval.py: Partial boundary hints with grid {}x{}.".format(FLAGS.dynamic_class_partial_boundary_hint_B, FLAGS.dynamic_class_partial_boundary_hint_B))
                print("eval.py: Drawing blocks with p {}.".format(FLAGS.dynamic_class_partial_boundary_hint_p))
                if FLAGS.dynamic_class_partial_boundary_full_block:
                    print("eval.py: Keeping entire block instead of masking boundaries.".format(FLAGS.boundary_threshold))
                else:
                    print("eval.py: Masking with boundary threshold {}.".format(FLAGS.boundary_threshold))

                print("----")

                if FLAGS.dynamic_class_partial_boundary_full_block:
                    boundary_mask = tf.cast(
                        tf.ones_like(samples[common.LABEL]), tf.uint8)
                else:
                    boundary_mask = tf.cast(
                        tf.less(samples[common.BOUNDARY_DMAP], FLAGS.boundary_threshold), tf.uint8)

                class_hints, hinted = tf.py_func(func=train_utils.generate_class_partial_boundaries_helper(B=FLAGS.dynamic_class_partial_boundary_hint_B,
                                                                                                        p=FLAGS.dynamic_class_partial_boundary_hint_p),
                                        inp=[samples[common.LABEL], boundary_mask],
                                        Tout=[tf.uint8, tf.bool])
                samples[common.HINT] = class_hints
                samples[common.HINT].set_shape(samples[common.LABEL].get_shape().as_list())
                # Now preprocess this. Set the flag so that  the rest of the work will be done as usual.
                FLAGS.hint_types = ['class_hint']
            ###

            if 'dynamic_class_hint' in FLAGS.hint_types:
                assert len(FLAGS.hint_types) == 1, 'When using dynamic class hints, do not use other hint types!'
                print("----")
                print("eval.py: Drawing hints with geo mean {}.".format(FLAGS.dynamic_class_hint_geo_mean))
                print("eval.py: Masking with boundary threshold {}.".format(FLAGS.boundary_threshold))
                print("----")
                boundary_mask = tf.cast(
                    tf.less(samples[common.BOUNDARY_DMAP], FLAGS.boundary_threshold), tf.uint8)
                class_hints, hinted = tf.py_func(func=train_utils.generate_class_clicks_helper(geo_mean=FLAGS.dynamic_class_hint_geo_mean),
                                        inp=[samples[common.LABEL], boundary_mask],
                                        Tout=[tf.uint8, tf.bool])
                samples[common.HINT] = class_hints
                samples[common.HINT].set_shape(samples[common.LABEL].get_shape().as_list())
                # Now preprocess this. Set the flag so that  the rest of the work will be done as usual.
                FLAGS.hint_types = ['class_hint']

            # If using class hints, preprocess into num_class binary mask channels
            if 'class_hint' in FLAGS.hint_types:
                assert len(FLAGS.hint_types) == 1, 'When using class hints, do not use other hint types!'
                num_classes = dataset.num_classes
                print('eval.py: num classes is {}'.format(num_classes))
                class_hint_channels_list = []
                for label in range(num_classes):
                    # Multiply by 255 is to bring into same range as image pixels...,
                    # and so feature_extractor mean subtraction will reduce it back to 0,1 range
                    class_hint_channel = tf.to_float(tf.equal(samples[common.HINT], label)) * 255
                    class_hint_channels_list.append(class_hint_channel)
                class_hint_channels = tf.concat(class_hint_channels_list, axis=-1)
                samples[common.HINT] = class_hint_channels

            # Get hints and concat to image as input into network
            samples[common.HINT] = tf.identity(
                samples[common.HINT], name=common.HINT)
            model_inputs = tf.concat(
                [samples[common.IMAGE], tf.to_float(samples[common.HINT])], axis=-1)
        else:
            # Just image is input into network
            model_inputs = samples[common.IMAGE]
        print('eval.py: shape {}'.format( model_inputs.get_shape().as_list()))

        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
            crop_size=FLAGS.eval_crop_size,
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride)

        if tuple(FLAGS.eval_scales) == (1.0,):
            tf.logging.info('Performing single-scale test.')
            predictions = model.predict_labels(
                # samples[common.IMAGE],
                model_inputs,
                model_options=model_options,
                image_pyramid=FLAGS.image_pyramid)
        else:
            tf.logging.info('Performing multi-scale test.')
            predictions = model.predict_labels_multi_scale(
                # samples[common.IMAGE],
                model_inputs,
                model_options=model_options,
                eval_scales=FLAGS.eval_scales,
                add_flipped_images=FLAGS.add_flipped_images)
        predictions = predictions[common.OUTPUT_TYPE]

        #predictions = tf.Print(predictions, predictions.get_shape())
        #from utils import train_utils
        #gen_boundaries = train_utils.generate_boundaries_helper(pixel_shift=1, ignore_label=255, distance_map=True, distance_map_scale=100, set_ignore_regions_to_ignore_value=False)
        #prediction_distance_map, _ = gen_boundaries(tf.to_float(tf.reshape(predictions, [1,1025, 2049])))
        #label_distance_map, _ = gen_boundaries(tf.to_float(tf.reshape(labels, [1, 1025, 2049])))

        predictions = tf.reshape(predictions, shape=[-1])
        labels = tf.reshape(samples[common.LABEL], shape=[-1])
        weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))

        # Set ignore_label regions to label 0, because metrics.mean_iou requires
        # range of labels = [0, dataset.num_classes). Note the ignore_label regions
        # are not evaluated since the corresponding regions contain weights =
        # 0.
        labels = tf.where(
            tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

        predictions_tag = 'miou'
        for eval_scale in FLAGS.eval_scales:
            predictions_tag += '_' + str(eval_scale)
        if FLAGS.add_flipped_images:
            predictions_tag += '_flipped'

        # Define the evaluation metric.
        metric_map = {}

        # mean iou
        metric_map[predictions_tag] = tf.metrics.mean_iou(
            predictions, labels, dataset.num_classes, weights=weights)

        # boundary distancemap l2
        #metric_map['boundary distance L2'] = tf.metrics.mean_squared_error(
            #prediction_distance_map, label_distance_map)


        metrics_to_values, metrics_to_updates = (
            tf.contrib.metrics.aggregate_metric_map(metric_map))

        for metric_name, metric_value in six.iteritems(metrics_to_values):
            slim.summaries.add_scalar_summary(
                metric_value, metric_name, print_summary=True)

        num_batches = int(
            math.ceil(dataset.num_samples / float(FLAGS.eval_batch_size)))

        tf.logging.info('Eval num images %d', dataset.num_samples)
        tf.logging.info('Eval batch size %d and num batch %d',
                        FLAGS.eval_batch_size, num_batches)

        num_eval_iters = None
        if FLAGS.max_number_of_evaluations > 0:
            num_eval_iters = FLAGS.max_number_of_evaluations
        config = tf.ConfigProto(
                                intra_op_parallelism_threads=16,
                                inter_op_parallelism_threads=1)
        slim.evaluation.evaluation_loop(
            master=FLAGS.master,
            checkpoint_dir=FLAGS.checkpoint_dir,
            logdir=FLAGS.eval_logdir,
            num_evals=num_batches,
            eval_op=list(metrics_to_updates.values()),
            max_number_of_evaluations=num_eval_iters,
            eval_interval_secs=FLAGS.eval_interval_secs,
            session_config=config)


if __name__ == '__main__':
    flags.mark_flag_as_required('checkpoint_dir')
    flags.mark_flag_as_required('eval_logdir')
    flags.mark_flag_as_required('dataset_dir')
    tf.app.run()
