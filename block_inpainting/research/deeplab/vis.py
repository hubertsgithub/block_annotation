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

"""Segmentation results visualization on a given set of images.

See model.py for more details and usage.
"""
import math
import os.path
import time
import numpy as np
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import save_annotation
import tqdm
import time
from deeplab.utils import train_utils

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('vis_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for visualizing the model.

flags.DEFINE_integer('vis_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_integer('vis_num_batches', 0,
                     'The number of batches to process. If set to <= 0 (default), then dataset.num_samples is used.')

flags.DEFINE_integer('also_vis_first_N', 0,
                     'also visualize the first N (default N=0) images. Useful for producing logits for seed images.')

flags.DEFINE_integer('start_idx', 0,
                     'vis start idx')

flags.DEFINE_bool('shuffle', False,
                     'shuffle images before visualizing')

flags.DEFINE_bool('overwrite_vis', False,
                     'overwrite existing visualizations')

flags.DEFINE_integer('shuffle_seed', 1234,
                     'shuffle seed')

flags.DEFINE_multi_integer('vis_crop_size', [513, 513],
                           'Crop size [height, width] for visualization.')

flags.DEFINE_multi_integer('vis_placeholder_size', [1, 1025, 2049, 19],
                           'softmax accumulation placeholder size.')

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

flags.DEFINE_string('vis_split', 'val',
                    'Which split of the dataset used for visualizing results')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_enum('colormap_type', 'pascal', ['pascal', 'cityscapes', 'ade20k', 'mscoco_panoptic', 'minc'],
                  'Visualization colormap type.')

flags.DEFINE_boolean('also_save_raw_predictions', False,
                     'Also save raw predictions.')

flags.DEFINE_boolean('save_logits', False,
                     '')

flags.DEFINE_boolean('compute_uncertainty', False,
                     'Compute uncertainty with MC dropout.'
                     'Use num_forward_passes iterations.')

flags.DEFINE_integer('compute_uncertainty_iterations', 5,
                     'Number of forward passes to compute uncertainty')

flags.DEFINE_boolean('convert_to_eval_id', True,
                     'Convert raw predictions to eval id (if applicable).')

flags.DEFINE_integer('max_number_of_iterations', 0,
                     'Maximum number of visualization iterations. Will loop '
                     'indefinitely upon nonpositive values.')



# The folder where semantic segmentation predictions are saved.
_SEMANTIC_PREDICTION_SAVE_FOLDER = 'segmentation_results'

# The folder where raw semantic segmentation predictions are saved.
_RAW_SEMANTIC_PREDICTION_SAVE_FOLDER = 'raw_segmentation_results'

# The format to save image.
_IMAGE_FORMAT = '%06d_image'

# The format to save prediction
_PREDICTION_FORMAT = '%06d_prediction'

# To evaluate Cityscapes results on the evaluation server, the labels used
# during training should be mapped to the labels for evaluation.
_CITYSCAPES_TRAIN_ID_TO_EVAL_ID = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22,
                                   23, 24, 25, 26, 27, 28, 31, 32, 33]


def _convert_train_id_to_eval_id(prediction, train_id_to_eval_id):
    """Converts the predicted label for evaluation.

    There are cases where the training labels are not equal to the evaluation
    labels. This function is used to perform the conversion so that we could
    evaluate the results on the evaluation server.

    Args:
      prediction: Semantic segmentation prediction.
      train_id_to_eval_id: A list mapping from train id to evaluation id.

    Returns:
      Semantic segmentation prediction whose labels have been changed.
    """
    converted_prediction = prediction.copy()
    for train_id, eval_id in enumerate(train_id_to_eval_id):
        converted_prediction[prediction == train_id] = eval_id

    return converted_prediction


def _process_batch(sess, original_images, semantic_predictions, image_names,
                   image_heights, image_widths, image_id_offset, save_dir,
                   raw_save_dir, logit_save_dir, uncertainty_save_dir,
                   train_id_to_eval_id=None,
                   save_logits=False,
                   logits=None,
                   fixed_features=None,
                   extra_to_run={},
                   samples_placeholders=None,
                   samples_orig=None,
                   compute_uncertainty=False,
                   num_forward_passes=1):
    """Evaluates one single batch qualitatively.

    Args:
      sess: TensorFlow session.
      original_images: One batch of original images.
      semantic_predictions: One batch of semantic segmentation predictions.
      image_names: Image names.
      image_heights: Image heights.
      image_widths: Image widths.
      image_id_offset: Image id offset for indexing images.
      save_dir: The directory where the predictions will be saved.
      raw_save_dir: The directory where the raw predictions will be saved.
      train_id_to_eval_id: A list mapping from train id to eval id.

      Input generator and network are decoupled so that multiple passes through the network can
        made with the same batch. Useful for MC dropout (computing uncertainty).
      samples_orig: run this to get a new batch from the input generator
      samples_placeholders: set the tensor values to data in samples_orig batch.
                            predictions, etc are run with samples_placeholders as the input.

      To compute uncertainty by only running later layers of network...
      fixed_features: tensor of fixed features from network which only needs to be computed once
      extra_to_run[accumulated_softmax, accumulated_softmax_sq]: accumulated softmax from fixed feature logits. Requires samples_placeholders['fixed_features'] to be set.
    """
    ####
    def softmax(arr):
        return np.exp(arr) / (np.sum(np.exp(arr), axis=-1)[...,np.newaxis])
    ####

    # Run samples_orig to get samples
    samples = sess.run([samples_orig])[0]

    # Map placeholders values to samples
    feed_dict = {v: samples[k] for k,v in samples_placeholders.items() if k in samples.keys()}

    # Skip processing if already processed.
    image_names_ = sess.run([image_names], feed_dict=feed_dict)[0]
    assert len(image_names_) == 1
    image_filename = os.path.basename(image_names_[0]).split('.png')[0].split('.jpg')[0]
    savename = os.path.join(save_dir, image_filename+".png")
    if os.path.exists(savename) and not FLAGS.overwrite_vis:
        print("  {} exists.".format(savename))
        print(">>> SKIPPING <<<<")
        return

    # Build tensors run list
    to_run = [original_images, image_names,
              image_heights, image_widths]

    semantic_predictions_idx = None
    if semantic_predictions is not None:
        semantic_predictions_idx = len(to_run)
        to_run.append(semantic_predictions)
    logits_idx = None
    if logits is not None:
        logits_idx = len(to_run)
        to_run.append(logits)
    fixed_features_idx = None
    if fixed_features is not None:
        fixed_features_idx = len(to_run)
        to_run.append(fixed_features)

    run_output = sess.run(to_run, feed_dict=feed_dict)

    # Gather run outputs
    original_images = run_output[0]
    image_names = run_output[1]
    image_heights = run_output[2]
    image_widths = run_output[3]
    if semantic_predictions is not None:
        semantic_predictions = run_output[semantic_predictions_idx]
    if logits is not None:
        logits_i = run_output[logits_idx]
    if fixed_features is not None:
        fixed_features = run_output[fixed_features_idx]



    # pred mean softmax_logits is mu = sum(softmax_logits, axis=newaxis) / num_forward_passes
    # pred var semantic_pred is sigma = sqrt(1 / (num_forward_passes - 1)) sqrt(
    #                          sum(softmax_logits**2, axis=newaxis)
    #                           - 2*mu*sum(softmax_logits, axis=newaxis)
    #                           + num_forward_passes * mu**2)
    # sum( (y - mu)^2 ) = sum(y**2) - 2mu*sum(y) + 2*N*mu**2
    if compute_uncertainty:

        if fixed_features is not None:
            feed_dict = {samples_placeholders['fixed_features']: fixed_features,
                         samples_placeholders['image']: samples['image'],
                         #samples_placeholders['accumulated_softmax']: np.zeros((1,1025, 2049, 19)),
                         #samples_placeholders['accumulated_softmax_sq']: np.zeros((1,1025, 2049, 19)),
                         #samples_placeholders['accumulated_softmax']: np.zeros((1,513, 513, 23)),
                         #samples_placeholders['accumulated_softmax_sq']: np.zeros((1,513, 513, 23)),
                         samples_placeholders['accumulated_softmax']: np.zeros(FLAGS.vis_placeholder_size),
                         samples_placeholders['accumulated_softmax_sq']: np.zeros(FLAGS.vis_placeholder_size),
                         }

            for i in tqdm.tqdm(range(num_forward_passes)):
                (accumulated_softmax,
                 accumulated_softmax_sq) = sess.run([extra_to_run['accumulated_softmax'],
                                                                   extra_to_run['accumulated_softmax_sq']],
                                                    feed_dict=feed_dict)
                feed_dict[samples_placeholders['accumulated_softmax']] = accumulated_softmax
                feed_dict[samples_placeholders['accumulated_softmax_sq']] = accumulated_softmax_sq

        else:

            assert not save_logits, "Do not save logits when computing uncertainty."
            assert logits is not None, "Logits are required to compute uncertainty."

            feed_dict.update({
                            #samples_placeholders['accumulated_softmax']: np.zeros((1,1025, 2049, 19)),
                            #samples_placeholders['accumulated_softmax_sq']: np.zeros((1,1025, 2049, 19)),
                            #samples_placeholders['accumulated_softmax']: np.zeros((1,513, 513, 23)),
                            #samples_placeholders['accumulated_softmax_sq']: np.zeros((1,513, 513, 23)),
                            samples_placeholders['accumulated_softmax']: np.zeros(FLAGS.vis_placeholder_size),
                            samples_placeholders['accumulated_softmax_sq']: np.zeros(FLAGS.vis_placeholder_size),
                              })

            # Run forward passes and compute softmax mean and variances.
            # TODO: This is numerically unstable.
            print('    Accumulating {} forward passes to compute uncertainty...'.format(num_forward_passes))
            for i in tqdm.tqdm(range(num_forward_passes)):
                #start_time_log = time.time()
                (accumulated_softmax,
                 accumulated_softmax_sq) = sess.run([extra_to_run['accumulated_softmax'],
                                                                   extra_to_run['accumulated_softmax_sq']],
                                                    feed_dict=feed_dict)
                feed_dict[samples_placeholders['accumulated_softmax']] = accumulated_softmax
                feed_dict[samples_placeholders['accumulated_softmax_sq']] = accumulated_softmax_sq

        #start_time = time.time()
        pred_mean = 1.0*accumulated_softmax / num_forward_passes
        pred_var = accumulated_softmax_sq -\
                    2 * pred_mean * accumulated_softmax +\
                    num_forward_passes * pred_mean**2
        pred_var = np.abs(pred_var)  # deal with negative values due to precision error. These values be small (ie 0)
        pred_var = np.sqrt((1.0 / (num_forward_passes - 1)) * pred_var)

        #print('        Time elapsed for computing mean and var: {}'.format(time.time() - start_time))
        #print(pred_var)

        # Overwrite semantic_predictions with mean prediction
        print('      Computing pred_mean_argmax...')
        start_time = time.time()
        pred_mean_argmax = np.argmax(pred_mean, axis=-1)
        #semantic_predictions_orig = semantic_predictions
        semantic_predictions = pred_mean_argmax
        #print(np.where(semantic_predictions != semantic_predictions_orig))
        #assert np.all(semantic_predictions == semantic_predictions_orig)
        print('        Time elapsed: {}'.format(time.time() - start_time))

        print('      Computing pred_var corresponding to pred_mean_argmax...')
        start_time = time.time()
        prediction_variances = np.zeros_like(pred_mean_argmax, dtype=pred_var.dtype)
        for ii in range(pred_var.shape[0]):
            for jj in range(pred_var.shape[1]):
                for kk in range(pred_var.shape[2]):
                    prediction_variances[ii,jj,kk] = pred_var[ii,jj,kk,pred_mean_argmax[ii,jj,kk]]
        print('        Time elapsed: {}'.format(time.time() - start_time))
        assert prediction_variances.shape == semantic_predictions.shape
        #print(np.max(prediction_variance))
        #print(np.min(prediction_variance))

    else:
        if logits is not None:
            logits = logits_i
        else:
            pass

    ### BELOW IS UNCHANGED. ###

    num_image = semantic_predictions.shape[0]
    for i in range(num_image):

        original_image = np.squeeze(original_images[i])
        #image_height = original_image.shape[0]
        #image_width = original_image.shape[1]
        image_height = np.squeeze(image_heights[i])
        image_width = np.squeeze(image_widths[i])
        semantic_prediction = np.squeeze(semantic_predictions[i])
        crop_semantic_prediction = semantic_prediction[
            :image_height, :image_width]

        image_filename = os.path.basename(image_names[i]).split('.png')[0].split('.jpg')[0]
        if image_filename == '':
            # Save image.
            save_annotation.save_annotation(
                original_image, save_dir, _IMAGE_FORMAT % (image_id_offset + i),
                add_colormap=False)

            # Save prediction.
            save_annotation.save_annotation(
                crop_semantic_prediction, save_dir,
                _PREDICTION_FORMAT % (image_id_offset + i), add_colormap=True,
                colormap_type=FLAGS.colormap_type)
        else:
            # Save image.
            save_annotation.save_annotation(
                original_image, save_dir, image_filename,
                add_colormap=False)

            # Save prediction.
            save_annotation.save_annotation(
                crop_semantic_prediction, save_dir,
                image_filename+"_vis", add_colormap=True,
                colormap_type=FLAGS.colormap_type)

        if FLAGS.also_save_raw_predictions:
            #image_filename = os.path.basename(image_names[i])
            image_filename = os.path.basename(image_names[i]).split('.png')[0].split('.jpg')[0]

            if train_id_to_eval_id is not None:
                crop_semantic_prediction = _convert_train_id_to_eval_id(
                    crop_semantic_prediction,
                    train_id_to_eval_id)
            save_annotation.save_annotation(
                crop_semantic_prediction, raw_save_dir, image_filename,
                add_colormap=False)

        if FLAGS.save_logits:
            assert logits is not None
            image_filename = os.path.basename(image_names[i]).split('.png')[0].split('.jpg')[0]
            #print(type(logits))
            #print(logits.shape)
            #logits = np.array(logits)

            with tf.gfile.Open('%s/%s.npy' % (logit_save_dir, image_filename), mode='w') as f:
                np.save(f, logits)

        if FLAGS.compute_uncertainty:
            prediction_variances = prediction_variances.astype(np.float16)
            image_filename = os.path.basename(image_names[i]).split('.png')[0].split('.jpg')[0]

            prediction_variance = np.squeeze(prediction_variances[i])
            crop_prediction_variance = prediction_variance[
                                        :image_height, :image_width]

            with tf.gfile.Open('%s/%s.npy' % (uncertainty_save_dir, image_filename), mode='w') as f:
                np.save(f, crop_prediction_variance)


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Get dataset-dependent information.
    dataset = segmentation_dataset.get_dataset(
        FLAGS.dataset, FLAGS.vis_split,
        dataset_dir=FLAGS.dataset_dir,
        use_input_hints=FLAGS.input_hints,
        hint_types=FLAGS.hint_types)

    train_id_to_eval_id = None
    if dataset.name == segmentation_dataset.get_cityscapes_dataset_name() and FLAGS.convert_to_eval_id:
        tf.logging.info('Cityscapes requires converting train_id to eval_id.')
        train_id_to_eval_id = _CITYSCAPES_TRAIN_ID_TO_EVAL_ID

    # Prepare for visualization.
    tf.gfile.MakeDirs(FLAGS.vis_logdir)
    save_dir = os.path.join(FLAGS.vis_logdir, _SEMANTIC_PREDICTION_SAVE_FOLDER)
    tf.gfile.MakeDirs(save_dir)
    raw_save_dir = os.path.join(
        FLAGS.vis_logdir, _RAW_SEMANTIC_PREDICTION_SAVE_FOLDER)
    tf.gfile.MakeDirs(raw_save_dir)

    logit_save_dir = os.path.join(
        FLAGS.vis_logdir, 'logits')
    tf.gfile.MakeDirs(logit_save_dir)

    uncertainty_save_dir = os.path.join(
        FLAGS.vis_logdir, 'uncertainties')
    tf.gfile.MakeDirs(uncertainty_save_dir)

    tf.logging.info('Visualizing on %s set', FLAGS.vis_split)

    g = tf.Graph()
    with g.as_default():
        # Running samples_orig will grab a new batch
        samples_orig = input_generator.get(dataset,
                                      FLAGS.vis_crop_size,
                                      FLAGS.vis_batch_size,
                                      min_resize_value=FLAGS.min_resize_value,
                                      max_resize_value=FLAGS.max_resize_value,
                                      resize_factor=FLAGS.resize_factor,
                                      dataset_split=FLAGS.vis_split,
                                      is_training=False,
                                      model_variant=FLAGS.model_variant)
        # samples_placeholders will represent a batch of data for the network. The values will be filled
        # by samples_orig. Decoupled so that same batch can be run through network multiple times.
        # See _process_batch.
        samples_placeholders = {}
        for k,v in samples_orig.items():
            samples_placeholders[k] = tf.placeholder(dtype=v.dtype, shape=v.shape, name='samples_{}'.format(k))

        # Since original code was written with 'samples' variable, leave original code alone and initialize samples dictionary here
        # The reason we don't use samples = samples_placeholders is because samples is overwritten several times
        # and we need to keep samples_placeholders in its original state in order to fill it with values from samples_orig.
        samples = {k: v for k,v in samples_placeholders.items()}

        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
            crop_size=FLAGS.vis_crop_size,
            atrous_rates=FLAGS.atrous_rates,
            output_stride=FLAGS.output_stride)

        if FLAGS.input_hints:  # or if common.HINT in samples.keys():
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
                print("WARNING: Do not use dynamic class hints when simulating crowdsourced points as the points should not change between runs.")
                print("vis.py: Drawing hints with geo mean {}.".format(FLAGS.dynamic_class_hint_geo_mean))
                print("vis.py: Masking with boundary threshold {}.".format(FLAGS.boundary_threshold))
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
                print('vis.py: num classes is {}'.format(num_classes))
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

        outputs_to_scales_to_logits = None
        logits = None
        predictions = None
        fixed_features = None
        extra_to_run = {}
        if tuple(FLAGS.eval_scales) == (1.0,):
            tf.logging.info('Performing single-scale test.')
            if FLAGS.compute_uncertainty and FLAGS.force_dropout_only_branch:
                fixed_features =  model._get_features_after_decoder(
                                        images=model_inputs,
                                        model_options=model_options,
                                        reuse=None,
                                        is_training=False,
                                        fine_tune_batch_norm=False,
                                        force_dropout=True,
                                        force_dropout_only_branch=FLAGS.force_dropout_only_branch,
                                        keep_prob=FLAGS.keep_prob)

                samples_placeholders['fixed_features'] = tf.placeholder(dtype=fixed_features.dtype,
                                                                        shape=fixed_features.shape)
                logits_from_fixed_features = model._get_branch_logits(
                                                        samples_placeholders['fixed_features'],
                                                        model_options.outputs_to_num_classes[common.OUTPUT_TYPE],
                                                        model_options.atrous_rates,
                                                        aspp_with_batch_norm=model_options.aspp_with_batch_norm,
                                                        kernel_size=model_options.logits_kernel_size,
                                                        reuse=None,
                                                        scope_suffix=common.OUTPUT_TYPE,
                                                        keep_prob=FLAGS.keep_prob,
                                                        force_dropout=True)
                logits_from_fixed_features = tf.image.resize_bilinear(logits_from_fixed_features,
                                                                      size=tf.shape(samples[common.IMAGE])[1:3],
                                                                      align_corners=True)

                softmax_from_fixed_features = tf.nn.softmax(logits_from_fixed_features)

                samples_placeholders['accumulated_softmax'] = tf.placeholder(dtype=softmax_from_fixed_features.dtype,
                                                                             shape=FLAGS.vis_placeholder_size)
                                                                             #shape=[1, 1025, 2049, 19])
                                                                             #shape=[1, 513, 513, 23])
                samples_placeholders['accumulated_softmax_sq'] = tf.placeholder(dtype=softmax_from_fixed_features.dtype,
                                                                             shape=FLAGS.vis_placeholder_size)
                                                                             #shape=[1, 1025, 2049, 19])
                                                                             #shape=[1, 513, 513, 23])

                accumulated_softmax = samples_placeholders['accumulated_softmax'] + softmax_from_fixed_features
                accumulated_softmax_sq = samples_placeholders['accumulated_softmax_sq'] + tf.square(softmax_from_fixed_features)
                extra_to_run['accumulated_softmax'] = accumulated_softmax
                extra_to_run['accumulated_softmax_sq'] = accumulated_softmax_sq

            elif FLAGS.save_logits or FLAGS.compute_uncertainty:
                predictions, outputs_to_scales_to_logits = model.predict_labels(
                    # samples[common.IMAGE],
                    model_inputs,
                    model_options=model_options,
                    image_pyramid=FLAGS.image_pyramid,
                    also_return_logits=True,
                    force_dropout=(FLAGS.compute_uncertainty or FLAGS.force_dropout),
                    force_dropout_only_branch=FLAGS.force_dropout_only_branch,
                    keep_prob=FLAGS.keep_prob)

                assert tuple(FLAGS.eval_scales) == (1.0,)
                assert len(outputs_to_scales_to_logits) == 1
                for output in sorted(outputs_to_scales_to_logits):
                    scales_to_logits = outputs_to_scales_to_logits[output]
                    logits = scales_to_logits[model._MERGED_LOGITS_SCOPE]

                if FLAGS.compute_uncertainty:
                    assert not FLAGS.save_logits
                    # We need full size logits to compute final predition and uncertainty.
                    logits = tf.image.resize_bilinear(logits,
                                            size=tf.shape(model_inputs)[1:3],
                                            align_corners=True)

                    softmax_logits = tf.nn.softmax(logits)

                    samples_placeholders['accumulated_softmax'] = tf.placeholder(dtype=softmax_logits.dtype,
                                                                                shape=FLAGS.vis_placeholder_size)
                                                                                #shape=[1, 1025, 2049, 19])
                                                                                #shape=[1, 513, 513, 23])
                    samples_placeholders['accumulated_softmax_sq'] = tf.placeholder(dtype=softmax_logits.dtype,
                                                                                shape=FLAGS.vis_placeholder_size)
                                                                                #shape=[1, 1025, 2049, 19])
                                                                                #shape=[1, 513, 513, 23])

                    accumulated_softmax = samples_placeholders['accumulated_softmax'] + softmax_logits
                    accumulated_softmax_sq = samples_placeholders['accumulated_softmax_sq'] + tf.square(softmax_logits)
                    extra_to_run['accumulated_softmax'] = accumulated_softmax
                    extra_to_run['accumulated_softmax_sq'] = accumulated_softmax_sq
            else:
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
            if FLAGS.save_logits:
                raise NotImplementedError("Multiscale logits aren't saved")

        if predictions is not None:
            predictions = predictions[common.OUTPUT_TYPE]

        if FLAGS.min_resize_value and FLAGS.max_resize_value:
            if FLAGS.input_hints:
                #raise Exception("***Unclear if this will work with hints. Look over the code.")
                print("***Unclear if this will work with hints. Look over the code.")
            # Only support batch_size = 1, since we assume the dimensions of original
            # image after tf.squeeze is [height, width, 3].
            assert FLAGS.vis_batch_size == 1

            # Reverse the resizing and padding operations performed in preprocessing.
            # First, we slice the valid regions (i.e., remove padded region) and then
            # we reisze the predictions back.
            original_image = tf.squeeze(samples[common.ORIGINAL_IMAGE])
            original_image_shape = tf.shape(original_image)
            predictions = tf.slice(
                predictions,
                [0, 0, 0],
                [1, original_image_shape[0], original_image_shape[1]])
            resized_shape = tf.to_int32([tf.squeeze(samples[common.HEIGHT]),
                                         tf.squeeze(samples[common.WIDTH])])
            predictions = tf.squeeze(
                tf.image.resize_images(tf.expand_dims(predictions, 3),
                                       resized_shape,
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                       align_corners=True), 3)

        tf.train.get_or_create_global_step()
        saver = tf.train.Saver(slim.get_variables_to_restore())
        sv = tf.train.Supervisor(graph=g,
                                 logdir=FLAGS.vis_logdir,
                                 init_op=tf.global_variables_initializer(),
                                 summary_op=None,
                                 summary_writer=None,
                                 global_step=None,
                                 saver=saver)

        if FLAGS.vis_num_batches <= 0:
            num_batches = int(math.ceil(
                dataset.num_samples / float(FLAGS.vis_batch_size)))
        else:
            num_batches = FLAGS.vis_num_batches

            if FLAGS.shuffle:
                shuffled_idxs = range(dataset.num_samples)
                np.random.seed(FLAGS.shuffle_seed)
                np.random.shuffle(shuffled_idxs)
            else:
                shuffled_idxs = range(dataset.num_samples)
            idxs_to_keep = shuffled_idxs[FLAGS.start_idx:FLAGS.start_idx+FLAGS.vis_num_batches]

            if FLAGS.also_vis_first_N > 0:
                idxs_to_keep.extend(shuffled_idxs[0:FLAGS.also_vis_first_N])

            print(sorted(idxs_to_keep)[:10])
            print("There are {} indices to keep.".format(len(idxs_to_keep)))

            num_batches = int(math.ceil(
                dataset.num_samples / float(FLAGS.vis_batch_size)))

        last_checkpoint = None

        # Loop to visualize the results when new checkpoint is created.
        num_iters = 0
        while (FLAGS.max_number_of_iterations <= 0 or
               num_iters < FLAGS.max_number_of_iterations):
            num_iters += 1
            last_checkpoint = slim.evaluation.wait_for_new_checkpoint(
                FLAGS.checkpoint_dir, last_checkpoint)
            start = time.time()
            tf.logging.info(
                'Starting visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                             time.gmtime()))
            tf.logging.info('Visualizing with model %s', last_checkpoint)

            with sv.managed_session(FLAGS.master,
                                    start_standard_services=False) as sess:
                sv.start_queue_runners(sess)
                sv.saver.restore(sess, last_checkpoint)

                image_id_offset = 0
                for batch in range(num_batches):

                    if batch in idxs_to_keep:
                        tf.logging.info('Visualizing batch %d / %d',
                                        batch + 1, num_batches)
                        _process_batch(sess=sess,
                                    original_images=samples[
                                        common.ORIGINAL_IMAGE],
                                    semantic_predictions=predictions,
                                    image_names=samples[common.IMAGE_NAME],
                                    image_heights=samples[common.HEIGHT],
                                    image_widths=samples[common.WIDTH],
                                    image_id_offset=image_id_offset,
                                    save_dir=save_dir,
                                    raw_save_dir=raw_save_dir,
                                    save_logits=FLAGS.save_logits,
                                    logits=logits,
                                    fixed_features=fixed_features,
                                    extra_to_run=extra_to_run,
                                    logit_save_dir=logit_save_dir,
                                    uncertainty_save_dir=uncertainty_save_dir,
                                    train_id_to_eval_id=train_id_to_eval_id,
                                    samples_orig=samples_orig,
                                    samples_placeholders=samples_placeholders,
                                    compute_uncertainty=FLAGS.compute_uncertainty,
                                    num_forward_passes=FLAGS.compute_uncertainty_iterations)
                    else:
                        # Run batch generator to skip this batch
                        sess.run([samples_orig])
                    image_id_offset += FLAGS.vis_batch_size

            tf.logging.info(
                'Finished visualization at ' + time.strftime('%Y-%m-%d-%H:%M:%S',
                                                             time.gmtime()))
            time_to_next_eval = start + FLAGS.eval_interval_secs - time.time()
            if time_to_next_eval > 0 and num_iters < FLAGS.max_number_of_iterations:
                time.sleep(time_to_next_eval)


if __name__ == '__main__':
    flags.mark_flag_as_required('checkpoint_dir')
    flags.mark_flag_as_required('vis_logdir')
    flags.mark_flag_as_required('dataset_dir')
    tf.app.run()
