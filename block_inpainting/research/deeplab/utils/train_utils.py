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
"""Utility functions for training."""

import six
import numpy as np

import tensorflow as tf
from scipy.ndimage import distance_transform_edt

slim = tf.contrib.slim


def get_boundary_weight_mask(boundary_distance_map, boundary_threshold, verbose=True):
    '''
    boundary_distance_map: tensor; distance map with distances from precise boundaries (0 is boundary)
    threshold: float; threshold to split boundary and nonboundary points.
               All points with distance < threshold is boundary

    verbose: print(debug statements)

    return float32 weight mask tensor
    '''
    assert 0 <= boundary_threshold <= 255  # distances are capped at 255. Normalized distance map is 255 from left to right or top to bottom. But it is sqrt(2)*255 from corner to corner. See minc-segmentations-scripts.
    if verbose:
        ## Debug ##
        dmap_min = tf.reduce_min(boundary_distance_map)
        dmap_max = tf.reduce_max(boundary_distance_map)
        boundary_distance_map = tf.Print(boundary_distance_map,
                                         [], message="train_utils.py: =================================================")
        boundary_distance_map = tf.Print(boundary_distance_map,
                                         [dmap_min, dmap_max], message="train_utils.py: boundary map min and max are ")
        boundary_distance_map = tf.Print(boundary_distance_map,
                                         [], message="train_utils.py: boundary map threshold is {}".format(boundary_threshold))
        ############

    # Threshold distance map to create boundary and nonboundary points. Distance map are 0 at true boundaries.
    # Greater than threshold is nonboundary; less than is boundary.
    boundary_mask = tf.cast(
        tf.less(boundary_distance_map, boundary_threshold), tf.int8)
    if verbose:
        boundary_mask = tf.Print(
            boundary_mask, [boundary_mask], message="train_utils.py: boundary mask ")

    # Count number of boundary and nonboundary points
    num_boundary = tf.count_nonzero(boundary_mask)
    num_nonboundary = tf.count_nonzero(1 - boundary_mask)
    total = tf.add(num_boundary, num_nonboundary)
    if verbose:
        total = tf.Print(total, [num_boundary, num_nonboundary, total],
                         message="train_utils.py: num boundary / nonboundary / total ")

    # Weigh boundary points by (total / (2 *  num_boundary))
    # Weigh nonboundary points by total / (2 * num_nonboundary)

    half_total = tf.truediv(tf.cast(total, dtype=tf.float32),
                            tf.constant(2.0, dtype=tf.float32))
    #half_total = tf.truediv(total, 2.0)
    boundary_weight = tf.truediv(half_total,
                                 tf.cast(num_boundary, dtype=tf.float32))
    nonboundary_weight = tf.truediv(half_total,
                                    tf.cast(num_nonboundary, dtype=tf.float32))

    # If boundary weight is infinity, then this means there are no boundary points.
    # Just set it to 0 to be safe.
    # In this case, nonboundary weight should be 1.
    boundary_weight = tf.where(tf.is_inf(boundary_weight),
                               tf.constant(0.0, dtype=tf.float32),
                               boundary_weight)
    nonboundary_weight = tf.where(tf.equal(boundary_weight, 0.0),
                                  tf.constant(1.0),
                                  nonboundary_weight)

    # If nonboundary weight is infinity, then this means there are no nonboundary points.
    # Just set it to 0 to be safe.
    # In this case, boundary weight should be 1.
    nonboundary_weight = tf.where(tf.is_inf(nonboundary_weight),
                                  tf.constant(0.0, dtype=tf.float32),
                                  nonboundary_weight)
    boundary_weight = tf.where(tf.equal(nonboundary_weight, 0.0),
                               tf.constant(1.0),
                               boundary_weight)

    if verbose:
        boundary_weight = tf.Print(boundary_weight, [boundary_weight, nonboundary_weight, half_total],
                                   message="train_utils.py: boundary_weight / nonboundary_weight, half_total")

    # Compute final boundary weight mask: nonboundary_weight * nonboundary + boundary_weight * boundary
    # When #nonboundary == #boundary, the weights will equal 1.
    nonboundaries = tf.cast(tf.equal(boundary_mask, 0), tf.float32)
    boundaries = tf.cast(tf.equal(boundary_mask, 1), tf.float32)
    boundary_weight_mask = tf.add(tf.multiply(nonboundary_weight, nonboundaries),
                                  tf.multiply(boundary_weight, boundaries))
    if verbose:
        boundary_weight_mask = tf.Print(boundary_weight_mask, [
                                        boundary_weight_mask], message="train_utils.py: boundary_weight_mask ")

    return boundary_weight_mask


def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  num_classes,
                                                  ignore_label,
                                                  loss_weight=1.0,
                                                  upsample_logits=True,
                                                  scope=None,
                                                  boundary_dmap=None,
                                                  boundary_threshold=None,
                                                  boundary_mask_verbose=True,
                                                  teacher_logits=None,
                                                  softmax_temperature=1.0,
                                                  loss_probability_threshold=1.0,
                                                  use_hard_label_bootstrap=False,
                                                  ):
    """Adds softmax cross entropy loss for logits of each scale.

    Args:
      scales_to_logits: A map from logits names for different scales to logits.
        The logits have shape [batch, logits_height, logits_width, num_classes].
      labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
      num_classes: Integer, number of target classes.
      ignore_label: Integer, label to ignore.
      loss_weight: Float, loss weight. OR list corresponding to each class.
      upsample_logits: Boolean, upsample logits or not.
      scope: String, the scope for the loss.

      probability_threshold: if softmax probability > threshold, don't compute loss (or gradient) for pixel

    Raises:
      ValueError: Label or logits is None.
    """

    if type(loss_weight) is not float \
        and not (type(loss_weight) is list and
                 len(loss_weight) == num_classes and
                 all(type(l) is float for l in loss_weight)):
        raise Exception('Loss weight should be a float or ' +
                        'a list of length num_classes of weights. ' +
                        'Set this in segmentation_dataset.py ' +
                        '(if not set, default is 1.0).')

    if labels is None:
        raise ValueError('No label for softmax cross entropy loss.')

    for scale, logits in six.iteritems(scales_to_logits):
        loss_scope = None
        if scope:
            loss_scope = '%s_%s' % (scope, scale)

        if upsample_logits:
            # Label is not downsampled, and instead we upsample logits.
            logits = tf.image.resize_bilinear(
                logits, tf.shape(labels)[1:3], align_corners=True)

            scaled_labels = labels
        else:
            # Label is downsampled to the same size as logits.
            scaled_labels = tf.image.resize_nearest_neighbor(
                labels, tf.shape(logits)[1:3], align_corners=True)

        # flatten labels.
        scaled_labels = tf.reshape(scaled_labels, shape=[-1])

        # Weigh labels by class weight, and set weight for ignore labels to 0.
        if type(loss_weight) is float:
            print('train_utils.py: Scaling labels with loss weight {}'.format(loss_weight))
            not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
                                                       ignore_label)) * loss_weight
        elif type(loss_weight) is list:
            # weights are zero by default.
            not_ignore_mask = tf.zeros_like(scaled_labels, dtype=tf.float32)

            # set the weight for each label
            for label, lw in enumerate(loss_weight):
                print('train_utils.py: Scaling label {} with loss weight {}'.format(label, lw))
                not_ignore_mask = tf.add(not_ignore_mask,
                                         tf.to_float(tf.equal(scaled_labels, label)) * lw)

            # ignore label is already set to zero by default but set it
            # explicitly to be safe.
            not_ignore_mask = tf.multiply(not_ignore_mask,
                                          tf.to_float(tf.not_equal(scaled_labels, ignore_label)))
        else:
            raise Exception("Loss weight must be a float or a list of floats.")

        # Weigh labels by boundary / nonboundary weight
        # Function: normalized_distancemap_tensor, distance_threshold --> weight map
        # not_ignore_mask *= foreground/background_weight_mask
        if boundary_dmap is not None:
            # flatten boundary_dmap to match labels
            boundary_dmap = tf.reshape(boundary_dmap, shape=[-1])
            assert boundary_threshold is not None
            print("train_utils.py: boundary distance maps exists. Weighing interior and boundary points. Threshold is set to {}".format(boundary_threshold))

            boundary_weight_mask = get_boundary_weight_mask(boundary_distance_map=boundary_dmap,
                                                            boundary_threshold=boundary_threshold,
                                                            verbose=boundary_mask_verbose)
            not_ignore_mask = tf.multiply(
                not_ignore_mask, boundary_weight_mask)

        ## Don't compute loss (or gradient) for examples where probability > threshold
        assert loss_probability_threshold > 0 and loss_probability_threshold <= 1.0
        if loss_probability_threshold < 1.0:
            print("train_utils.py: NOT COMPUTING LOSS IF SOFTMAX PROB > {}".format(loss_probability_threshold))
            print("train_utils.py: Softmax prob computed with logits scaled by temperature {}".format(softmax_temperature))

            softmax = tf.nn.softmax(logits=tf.reshape(logits / softmax_temperature,
                                                      shape=[-1, num_classes]))
            max_softmax_prob = tf.reduce_max(softmax, axis=-1)
            max_softmax_prob_mask = tf.to_float(tf.less(max_softmax_prob, loss_probability_threshold))
            not_ignore_mask = tf.multiply(not_ignore_mask, max_softmax_prob_mask)

        one_hot_labels = slim.one_hot_encoding(
            scaled_labels, num_classes, on_value=1.0, off_value=0.0)

        ## Add hard label bootstrap loss (self-predicted labels)
        if use_hard_label_bootstrap:
            print("train_utils.py: Hard label bootstrapping with weight 0.2")
            one_hot_predicted_labels = slim.one_hot_encoding(
                tf.argmax(tf.nn.softmax(tf.reshape(logits, shape=[-1, num_classes])), axis=-1),
                num_classes,
                on_value=1.0,
                off_value=0.0)
            # Add loss with 0.2 weight
            tf.losses.softmax_cross_entropy(
                one_hot_predicted_labels,
                tf.reshape(logits, shape=[-1, num_classes]),
                weights=not_ignore_mask * 0.2,
                scope=loss_scope)
            # Reweigh loss against annotated labels to 0.8
            not_ignore_mask *= 0.8


        tf.losses.softmax_cross_entropy(
            one_hot_labels,
            tf.reshape(logits, shape=[-1, num_classes]),
            weights=not_ignore_mask,
            scope=loss_scope)

        ## Add soft label loss against teacher logits.
        if teacher_logits is not None:
            teacher_logits = tf.image.resize_bilinear(
                teacher_logits, tf.shape(logits)[1:3], align_corners=True)

            logits = tf.truediv(logits, softmax_temperature)
            teacher_logits = tf.truediv(teacher_logits, softmax_temperature)
            teacher_softmax = tf.nn.softmax(logits=tf.reshape(teacher_logits,
                                                              shape=[-1, num_classes]))

            tf.losses.softmax_cross_entropy(
                onehot_labels=teacher_softmax,
                logits=tf.reshape(logits, shape=[-1, num_classes]),
                weights=not_ignore_mask * tf.square(softmax_temperature),
                scope=loss_scope)





def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      initialize_last_layer,
                      last_layers,
                      ignore_missing_vars=False):
    """Gets the function initializing model variables from a checkpoint.

    Args:
      train_logdir: Log directory for training.
      tf_initial_checkpoint: TensorFlow checkpoint for initialization.
      initialize_last_layer: Initialize last layer or not.
      last_layers: Last layers of the model.
      ignore_missing_vars: Ignore missing variables in the checkpoint.

    Returns:
      Initialization function.
    """
    if tf_initial_checkpoint is None:
        tf.logging.info('Not initializing the model from a checkpoint.')
        return None

    if tf.train.latest_checkpoint(train_logdir):
        tf.logging.info('Ignoring initialization; other checkpoint exists')
        return None

    tf.logging.info('Initializing model from path: %s', tf_initial_checkpoint)

    # Variables that will not be restored.
    exclude_list = ['global_step']
    if not initialize_last_layer:
        exclude_list.extend(last_layers)

    variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)

    return slim.assign_from_checkpoint_fn(
        tf_initial_checkpoint,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)


def get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier):
    """Gets the gradient multipliers.

    The gradient multipliers will adjust the learning rates for model
    variables. For the task of semantic segmentation, the models are
    usually fine-tuned from the models trained on the task of image
    classification. To fine-tune the models, we usually set larger (e.g.,
    10 times larger) learning rate for the parameters of last layer.

    Args:
      last_layers: Scopes of last layers.
      last_layer_gradient_multiplier: The gradient multiplier for last layers.

    Returns:
      The gradient multiplier map with variables as key, and multipliers as value.
    """
    gradient_multipliers = {}

    for var in slim.get_model_variables():
        # Double the learning rate for biases.
        if 'biases' in var.op.name:
            gradient_multipliers[var.op.name] = 2.

        # Use larger learning rate for last layer variables.
        for layer in last_layers:
            if layer in var.op.name and 'biases' in var.op.name:
                gradient_multipliers[var.op.name] = 2 * \
                    last_layer_gradient_multiplier
                break
            elif layer in var.op.name:
                gradient_multipliers[
                    var.op.name] = last_layer_gradient_multiplier
                break

    return gradient_multipliers


def get_model_learning_rate(
        learning_policy, base_learning_rate, learning_rate_decay_step,
        learning_rate_decay_factor, training_number_of_steps, learning_power,
        slow_start_step, slow_start_learning_rate):
    """Gets model's learning rate.

    Computes the model's learning rate for different learning policy.
    Right now, only "step" and "poly" are supported.
    (1) The learning policy for "step" is computed as follows:
      current_learning_rate = base_learning_rate *
        learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
    See tf.train.exponential_decay for details.
    (2) The learning policy for "poly" is computed as follows:
      current_learning_rate = base_learning_rate *
        (1 - global_step / training_number_of_steps) ^ learning_power

    Args:
      learning_policy: Learning rate policy for training.
      base_learning_rate: The base learning rate for model training.
      learning_rate_decay_step: Decay the base learning rate at a fixed step.
      learning_rate_decay_factor: The rate to decay the base learning rate.
      training_number_of_steps: Number of steps for training.
      learning_power: Power used for 'poly' learning policy.
      slow_start_step: Training model with small learning rate for the first
        few steps.
      slow_start_learning_rate: The learning rate employed during slow start.

    Returns:
      Learning rate for the specified learning policy.

    Raises:
      ValueError: If learning policy is not recognized.
    """
    global_step = tf.train.get_or_create_global_step()
    if learning_policy == 'step':
        learning_rate = tf.train.exponential_decay(
            base_learning_rate,
            global_step,
            learning_rate_decay_step,
            learning_rate_decay_factor,
            staircase=True)
    elif learning_policy == 'poly':
        learning_rate = tf.train.polynomial_decay(
            base_learning_rate,
            global_step,
            training_number_of_steps,
            end_learning_rate=0,
            power=learning_power)
    else:
        raise ValueError('Unknown learning policy.')

    # Employ small learning rate at the first few steps for warm start.
    return tf.where(global_step < slow_start_step, slow_start_learning_rate,
                    learning_rate)

def get_model_inputs(samples, hints=False, class_hints=False, num_classes=-1):
    '''
    Helper function to concatenate images with hints to form model input.
    If hints are class hints, then additional processing is done to convert 1 channel hint to num_class binary channels.
    '''
    raise NotImplementedError()
    return model_inputs


# Custom train_step function that exits if training step reaches a multiple of N
# for validation purposes. Run evaluation loop manually and then run restart training loop manually
#from tensorflow.python.framework import constant_op


def train_step_exit(N=10000):
    ''' Exit every N steps (e.g. for validation)'''
    default_train_step_fn = slim.learning.train_step

    #####################################################################
    def train_step(sess, train_op, global_step, train_step_kwargs):

        # train op may be generated by slim.learning.create_train_op.
        #   train_op computes losses and applies gradients
        # However, in deeplab, it is generated in train.py
        #   See train_tensor (generated from total_loss)

        total_loss, should_stop = default_train_step_fn(
            sess, train_op, global_step, train_step_kwargs)
        # If step is a multiple of N, then stop.
        step = global_step.eval(session=sess)
        if step % N == 0:
            should_stop = True
        return total_loss, should_stop
    #####################################################################

    return train_step

import time
def train_step_custom(VALIDATION_N=10000, ACCUM_OP=None, ACCUM_STEPS=5):
    '''
    Exit every N steps (e.g. for validation)
    Run accum_op to accumulate gradients
    '''
    #default_train_step_fn = slim.learning.train_step # Use modified version below.

    #####################################################################
    def train_step(sess, train_op, global_step, train_step_kwargs):
        if not hasattr(train_step, "accum_counter"):
            train_step.accum_counter = 0
            train_step.start_time = time.time()

        if train_step.accum_counter < ACCUM_STEPS:
            train_step.accum_counter += 1
            sess.run(ACCUM_OP)
            return None, False
        else:
            total_loss, should_stop = default_train_step_fn(
                sess, train_op, global_step, train_step_kwargs, start_time=train_step.start_time)

            # If step is a multiple of N, then stop.
            step = global_step.eval(session=sess)
            if step % VALIDATION_N == 0:
                should_stop = True

            train_step.accum_counter = 0
            train_step.start_time = time.time()
            return total_loss, should_stop
    #####################################################################

    return train_step

# Copied from tensorflow/tensorflow/contrib/slim/learning and modified slightly.
def default_train_step_fn(sess, train_op, global_step, train_step_kwargs, start_time=None):
    """Function that takes a gradient step and specifies whether to stop.

    Args:
      sess: The current session.
      train_op: An `Operation` that evaluates the gradients and returns the
        total loss.
      global_step: A `Tensor` representing the global training step.
      train_step_kwargs: A dictionary of keyword arguments.

    Returns:
      The total loss and a boolean indicating whether or not to stop training.

    Raises:
      ValueError: if 'should_trace' is in `train_step_kwargs` but `logdir` is not.
    """
    if start_time == None:
        start_time = time.time()

    trace_run_options = None
    run_metadata = None
    if 'should_trace' in train_step_kwargs:
        raise Exception("Shouldn't be used since train_step_custom does not use this when calling sess.run. Unsure if required, so flag this for now.")
        #if 'logdir' not in train_step_kwargs:
        #    raise ValueError('logdir must be present in train_step_kwargs when '
        #                     'should_trace is present')
        #if sess.run(train_step_kwargs['should_trace']):
        #    trace_run_options = config_pb2.RunOptions(
        #        trace_level=config_pb2.RunOptions.FULL_TRACE)
        #    run_metadata = config_pb2.RunMetadata()

    total_loss, np_global_step = sess.run([train_op, global_step],
                                          options=trace_run_options,
                                          run_metadata=run_metadata)
    time_elapsed = time.time() - start_time

    if run_metadata is not None:
        raise Exception("Shouldn't be used since train_step_custom does not use this when calling sess.run. Unsure if required, so flag this for now.")
        #tl = timeline.Timeline(run_metadata.step_stats)
        #trace = tl.generate_chrome_trace_format()
        #trace_filename = os.path.join(train_step_kwargs['logdir'],
        #                              'tf_trace-%d.json' % np_global_step)
        #logging.info('Writing trace to %s', trace_filename)
        #file_io.write_string_to_file(trace_filename, trace)
        #if 'summary_writer' in train_step_kwargs:
        #    train_step_kwargs['summary_writer'].add_run_metadata(run_metadata,
        #                                                         'run_metadata-%d' %
        #                                                         np_global_step)

    if 'should_log' in train_step_kwargs:
        if sess.run(train_step_kwargs['should_log']):
            tf.logging.info('global step %d: loss = %.4f (%.3f sec/step)',
                         np_global_step, total_loss, time_elapsed)

    if 'should_stop' in train_step_kwargs:
        should_stop = sess.run(train_step_kwargs['should_stop'])
    else:
        should_stop = False

    return total_loss, should_stop


def generate_class_clicks_tensorflow(input_label_map, mask=None, ignore_value=255, geo_mean=8):
    '''

    Tensorflow implementation of generate_class_clicks from minc-segmentations-scripts.

    input_label_map: NxHxW tensor with per pixel class labels
    mask: NxHxW 0,1 mask. Pixels in input_label_map with corresponding mask value of 0 are converted to ignore_value
    ignore_value: ignore class label.

    Return:
    output = {'class_hints': ..., 'hinted':....}
    class_hints: HxW array with class label at hinted pixels, and ignore_value at nonhinted pixels. If a hinted pixel belongs to ignore class or is masked out, that pixel will also have value ignore_value.
    hinted: True if non ignore_value hints exist in the hint map.

    The sample strategy is based on deep interactive colorization paper.
    '''

    label_arr = tf.cast(input_label_map, tf.int32)
    if mask is not None:
        # label_arr[mask == 0] = ignore_value
        #mask_0 = tf.cast(tf.equal(mask, 0), tf.uint8)
        #label_arr = mask_0 * ignore_value + label_arr * (1 - mask_0)
        label_arr = tf.where(tf.equal(mask, 0), tf.ones_like(label_arr, tf.int32) * ignore_value, label_arr)

    #hints = np.ones(label_arr.shape, dtype=np.uint8) * ignore_value  # Default hints are that no pixels are hinted
    hints = tf.ones_like(label_arr, dtype=tf.int32) * ignore_value  # Default hints are that no pixels are hinted

    assert label_arr.get_shape().as_list() == hints.get_shape().as_list()

    #num_ignore = np.sum(label_arr==ignore_value)
    #num_not_ignore = np.sum(label_arr!=ignore_value)
    #frac = 1.0*num_not_ignore / (num_ignore + num_not_ignore)
    num_ignore = tf.reduce_sum(tf.to_float(tf.equal(label_arr, ignore_value)))
    num_not_ignore = tf.reduce_sum(tf.to_float(tf.not_equal(label_arr, ignore_value)))
    frac = tf.truediv(num_not_ignore, num_ignore + num_not_ignore)

    # In paper, the number of points are drawn from geometric distribution with p=1/8
    # Then point location is sampled from 2D gaussian with mean image center and variance (D/4)**2 for both dims.
    # Revealed patch size is 1x1 to 9x9 uniformly drawn.
    # Instead of generating binary mask, I will mask on hints directly. Set ignored/unhinted regions to 255.

    #num_points = np.random.geometric(p=1.0/geo_mean)
    #num_points /= frac
    #num_points = int(np.ceil(num_points))
    num_points = tf.truediv(tf.constant(np.random.geometric(p=1.0/geo_mean), tf.float32), frac)
    num_points = tf.to_int32(tf.ceil(num_points))


    #mean = np.array([d/2 for d in label_arr.shape])
    #sigma = np.array([d**2/16 for d in label_arr.shape]) * np.eye(2)
    #locations = np.random.multivariate_normal(mean=mean, cov=sigma, size=num_points)
    mean = tf.to_float(tf.truediv(tf.constant(label_arr.get_shape().as_list()[1:3]), 2))
    sigma = tf.to_float(tf.truediv( tf.square(tf.constant(label_arr.get_shape().as_list()[1:3])), 16))
    batchsize = label_arr.get_shape().as_list()[0]
    locations = [tf.random_normal(shape=[batchsize, num_points], mean=mean[0], stddev=sigma[0]),
                tf.random_normal(shape=[batchsize, num_points], mean=mean[1], stddev=sigma[1])]

    def condition(i, num_points, locations, hints):
        return tf.less(i, num_points)

    def draw_patches(i, num_points, locations, hints):
        #x,y = (int(round(d)) for d in location)
        x = tf.round(locations[0][:,i])
        y = tf.round(locations[0][:,i])

        w,h = hints.get_shape().as_list()[1:3]
        w = tf.ones_like(x) * w
        h = tf.ones_like(y) * h
        zero = tf.zeros_like(x)

        patch_size = int(round(np.random.uniform(low=1, high=9)))

        patch_x_range = [ tf.reduce_max([x - tf.ceil((patch_size-1.0)/2), zero ], axis=0),
                          tf.reduce_min([x + tf.floor((patch_size-1.0)/2), w-1 ], axis=0) ]
        patch_x_range[0] = tf.to_int32(tf.reduce_max([ zero, tf.reduce_min([ w, patch_x_range[0] ], axis=0) ], axis=0))  # if <0, this sets it to 0. if > w, this sets it to w.
        patch_x_range[1] = tf.to_int32(tf.reduce_max([ zero, tf.reduce_min([ w, patch_x_range[1] ], axis=0) ], axis=0))  # if <0, this sets it to 0. if > w, this sets it to w.

        patch_y_range = [ tf.reduce_max([y - tf.ceil((patch_size-1.0)/2), zero ], axis=0),
                          tf.reduce_min([y + tf.floor((patch_size-1.0)/2), h-1 ], axis=0) ]
        patch_y_range[0] = tf.to_int32(tf.reduce_max([ zero, tf.reduce_min([ h, patch_y_range[0] ], axis=0) ], axis=0))  # if <0, this sets it to 0. if > h, this sets it to h.
        patch_y_range[1] = tf.to_int32(tf.reduce_max([ zero, tf.reduce_min([ h, patch_y_range[1] ], axis=0) ], axis=0))  # if <0, this sets it to 0. if > h, this sets it to h.

        try:
            n,w,h = hints.get_shape().as_list()[0:3]
            w_range = tf.range(w) * tf.to_int32(tf.ones([n, 1]))
            h_range = tf.range(h) * tf.to_int32(tf.ones([n, 1]))
            x_mask = tf.to_int32(tf.logical_and(tf.greater_equal(w_range, tf.reshape(patch_x_range[0], [n,1])),
                                    tf.less(w_range, tf.reshape(patch_x_range[1]+1, [n,1]))))
            y_mask = tf.to_int32(tf.logical_and(tf.greater_equal(h_range, tf.reshape(patch_y_range[0], [n,1])),
                                    tf.less(h_range, tf.reshape(patch_y_range[1]+1, [n,1]))))
            patch_mask = tf.reshape(x_mask, [n, w, 1, 1]) * tf.reshape(y_mask, [n, 1, h, 1])
            ## Conditions for assignment of hint (need both):
            # patch_mask[location] = 1 and hint[location] = ignore_value.
            # Recall that hints are initialized to ignore_value.
            # Therefore, to assign hint, add hint value and subtract ignore_value.
            # Add 0s to make nochange.
            sub_hints = tf.where( tf.logical_and(tf.equal(hints, ignore_value),
                                                 tf.equal(patch_mask, 1)),
                                  label_arr - ignore_value,
                                  tf.zeros_like(label_arr))
            hints = hints + sub_hints

        except Exception as e:
            print(e)
            import ipdb; ipdb.set_trace()

        return [i+1, num_points, locations, hints]

    while_loop_output = tf.while_loop(cond=condition,
                                    body=draw_patches,
                                    loop_vars=[0,num_points,locations,hints],
                                    parallel_iterations=100,
                                    swap_memory=False)

    hints = while_loop_output[-1]
    #hints = tf.Print(hints, [tf.reduce_all(tf.equal(hints, ignore_value))], message='hints all ignore: ')
    #hints = tf.Print(hints, [tf.unique(tf.reshape(hints, [-1]))], message='hints unique: ')

    return hints

### Python Numpy version ###
def generate_class_clicks_helper(ignore_value=255, geo_mean=300, min_frac=0.10):
    print("train_utils.py: GENERATE_CLASS_CLICKS_HELPER MIN_FRAC IS {}. Hints are not drawn if frac is below this threshold".format(min_frac))

    def generate_class_clicks(input_label_map, mask=None, ignore_value=ignore_value, geo_mean=geo_mean,min_frac=min_frac):
        '''
        input_label_map: NxHxW array with per pixel class labels
        mask: NxHxW 0,1 mask. Pixels in input_label_map with corresponding mask value of 0 are converted to ignore_value
        ignore_value: ignore class label.

        Return:
        output = {'class_hints': ..., 'hinted':....}
        class_hints: HxW array with class label at hinted pixels, and ignore_value at nonhinted pixels. If a hinted pixel belongs to ignore class or is masked out, that pixel will also have value ignore_value.
        hinted: True if non ignore_value hints exist in the hint map.

        min_frac: if frac < min frac, then draw 0 hints (i.e., if not_ignore to ignore ratio is too low, don't bother drawing hints because we would have to draw way too many to get very little useful hints. Typically this is true in case of sparsely labelled images)

        The sample strategy is based on deep interactive colorization paper.
        '''

        label_arr = np.array(input_label_map, dtype=np.uint8)
        if mask is not None:
            mask = np.array(mask, dtype=np.uint8)
            label_arr[mask == 0] = ignore_value
        hints = np.ones(label_arr.shape, dtype=np.uint8) * ignore_value  # Default hints are that no pixels are hinted

        assert label_arr.shape == hints.shape

        num_ignore = np.sum(label_arr==ignore_value)
        num_not_ignore = np.sum(label_arr!=ignore_value)
        frac = 1.0*num_not_ignore / (num_ignore + num_not_ignore)
        if frac == 0:
            print("FRAC IS ZERO!!!")
        if frac < min_frac:
            #print("frac too low (min frac = {}). Drawing zero points.".format(min_frac))
            frac = 0

        # In paper, the number of points are drawn from geometric distribution with p=1/8
        # Then point location is sampled from 2D gaussian with mean image center and variance (D/4)**2 for both dims.
        # Revealed patch size is 1x1 to 9x9 uniformly drawn.
        # Instead of generating binary mask, I will mask on hints directly. Set ignored/unhinted regions to 255.
        if frac > 0:
            num_points = np.random.geometric(p=1.0/geo_mean)
            num_points /= frac
        else:
            num_points = 0
        num_points = int(np.ceil(num_points))
        #print('  USING GEO_MEAN: {}'.format(geo_mean))
        #print('  Drawing {} hints...'.format(num_points))
        mean = np.array([d/2 for d in label_arr.shape[1:3]])
        sigma = np.array([d**2/16 for d in label_arr.shape[1:3]]) * np.eye(2)
        locations = np.random.multivariate_normal(mean=mean, cov=sigma, size=num_points)

        num_hinted = 0
        for location in locations:
            x,y = (int(round(d)) for d in location)

            # If location out of bounds, just skip it. Alternative is to bring it within bounds but this messes up sample strategy a bit.
            if x < 0 or y < 0:
                #print('Location out of bounds')
                continue
            if x > label_arr.shape[1] or y > label_arr.shape[2]:
                #print('Location out of bounds')
                continue

            patch_size = int(round(np.random.uniform(low=1, high=9)))
            patch_x_range = [ int(max(x - np.ceil((patch_size-1.0)/2), 0)), int(min(x + np.floor((patch_size-1.0)/2), label_arr.shape[1]-1)) ]
            patch_y_range = [ int(max(y - np.ceil((patch_size-1.0)/2), 0)), int(min(y + np.floor((patch_size-1.0)/2), label_arr.shape[2]-1)) ]

            try:
                hints[:, patch_x_range[0]:patch_x_range[1]+1, patch_y_range[0]:patch_y_range[1]+1] =\
                    label_arr[:, patch_x_range[0]:patch_x_range[1]+1, patch_y_range[0]:patch_y_range[1]+1]
            except Exception as e:
                print(e)
                import ipdb; ipdb.set_trace()

        class_hints = hints
        hinted = (hints!=ignore_value).any()

        return class_hints, hinted

    return generate_class_clicks

def generate_class_partial_boundaries_helper(ignore_value=255, B=10, p=0.5):

    def generate_class_partial_boundaries(input_label_map, mask=None, ignore_value=ignore_value, B=B, p=p):
        '''
        Divides image into B*B grid. Sample p grids. Hint all of the boundaries (as defined by mask) in the sampled grids.

        input_label_map: NxHxW array with per pixel class labels
        mask: NxHxW 0,1 mask. Pixels in input_label_map with corresponding mask value of 0 are converted to ignore_value
        ignore_value: ignore class label.

        Return:
        output = {'class_hints': ..., 'hinted':....}
        class_hints: HxW array with class label at hinted pixels, and ignore_value at nonhinted pixels. If a hinted pixel belongs to ignore class or is masked out, that pixel will also have value ignore_value.
        hinted: True if non ignore_value hints exist in the hint map.
        '''
        label_arr = np.array(input_label_map, dtype=np.uint8)
        if mask is not None:
            mask = np.array(mask, dtype=np.uint8)
            label_arr[mask == 0] = ignore_value
        hints = np.ones(label_arr.shape, dtype=np.uint8) * ignore_value  # Default hints are that no pixels are hinted
        assert label_arr.shape == hints.shape

        num_hinted = 0

        assert 0 < p <= 1
        num_to_sample = int(p * B * B)
        sampled_blocks = np.random.choice(range(B*B), num_to_sample, replace=False)

        assert len(label_arr.shape) == 4  #(NxHxWxC)
        block_x_size = label_arr.shape[1] // B
        block_y_size = label_arr.shape[2] // B
        assert block_x_size > 0
        assert block_y_size > 0

        for block_num in sampled_blocks:
            x_start = ((block_num % B) * block_x_size)
            x_end = x_start + block_x_size
            assert x_end <= label_arr.shape[1]

            y_start = ((block_num - (block_num % B)) // B * block_y_size)
            y_end = y_start + block_y_size
            assert y_end <= label_arr.shape[2]

            try:
                hints[:, x_start:x_end, y_start:y_end] =\
                    label_arr[:, x_start:x_end, y_start:y_end]
            except Exception as e:
                print(e)
                import ipdb; ipdb.set_trace()

        class_hints = hints
        hinted = (hints!=ignore_value).any()

        return class_hints, hinted

    return generate_class_partial_boundaries

def generate_boundaries_helper(pixel_shift=1, ignore_label=255, distance_map=True, distance_map_scale=100, set_ignore_regions_to_ignore_value=False):

    def generate_boundaries(input_array):
        if len(input_array.shape) == 3:
            input_array = input_array[...,np.newaxis]
        assert len(input_array.shape) == 4, input_array.shape
        assert input_array.shape[-1] == 1


        img = np.array(input_array, dtype=np.float32)

        # Get boundaries using element-wise multiplication.
        img_left_shift = np.array(img) #, dtype=np.uint8)
        img_left_shift[:, pixel_shift:,:] = img[:, :-pixel_shift,:]
        img_left_shift[:, 0:pixel_shift,:] = ignore_label
        img_up_shift = np.array(img) #, dtype=np.uint8)
        img_up_shift[:, :,pixel_shift:] = img[:, :,:-pixel_shift]
        img_up_shift[:, :,0:pixel_shift] = ignore_label

        # This works.
        img_and = img_left_shift == img
        img_and *= img_up_shift == img
        img_and = 1 - img_and

       # # Alt_img --> this is (very very nearly) equivalent
       # alt_img = 1 -\
       #     np.minimum(1, np.abs(np.concatenate([np.ones_like(img[:,0:pixel_shift,:])*ignore_label,
       #                 img[:,:-pixel_shift,:]], axis=1) - img))

       # alt_img *= 1 -\
       #     np.minimum(1, np.abs(np.concatenate([np.ones_like(img[:,:, 0:pixel_shift])*ignore_label,
       #                 img[:,:, :-pixel_shift]], axis=2) - img))
       # alt_img = 1 - alt_img
       # #print(np.count_nonzero(alt_img != img_and))
       # #assert np.all( alt_img == img_and )
       # img_and = alt_img
       # ##

        if ignore_label is not None:
            ### Remove labels against ignore_label
            ignore_idxs = np.logical_or((img_up_shift == ignore_label),\
                                    np.logical_or((img_left_shift == ignore_label), (img == ignore_label)))
            if set_ignore_regions_to_ignore_value:
                # Set ignore regions all to ignore_label
                img_and[ignore_idxs] = ignore_label
            else:
                # Set ignore regions to 0 (non boundary)
                img_and[ignore_idxs] = 0
            ###

        boundaries = img_and.astype(np.float32)

        weights = np.ones_like(boundaries)
        weights[boundaries == ignore_label] = 0

        num_boundaries = np.count_nonzero(boundaries==1)
        num_nonboundaries = np.count_nonzero(boundaries==0)
        weights[boundaries == 1] = 1.0 * (num_boundaries + num_nonboundaries) / (num_boundaries+1)
        weights[boundaries == 0] = 1.0 * (num_boundaries + num_nonboundaries) / (num_nonboundaries + 1)

        if distance_map:
            boundaries_distance = np.zeros_like(boundaries, np.float32)
            invert_boundaries = 1 - boundaries
            for batch_num in range(boundaries.shape[0]):
                boundaries_distance[batch_num,:,:,0] = distance_transform_edt(invert_boundaries[batch_num,:,:,0], sampling=(1.0/img_and.shape[1], 1.0/img_and.shape[2])) * distance_map_scale
            boundaries = boundaries_distance

        return boundaries, weights

    return generate_boundaries

if __name__ == '__main__':
    print("DEBUG CODE")
    import PIL.Image
    f = generate_boundaries_helper(distance_map=False,
                                   distance_map_scale = 100./255,
                                   pixel_shift=2)
    input_labels = "/home/hlin/projects/datasets/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png"
    img = PIL.Image.open(input_labels).convert("L")
    img = np.array(img, dtype=np.uint8)
    img = img[...,np.newaxis]
    #img = np.array([img, np.random.rand(*img.shape)*1000])
    img = np.array([img])
    img, weights= f(img)
    img = img * 255
    print(img.dtype)
    print(img.shape)
    print(np.max(img[0]))
    img = img.astype(np.uint8)
    PIL.Image.fromarray(img[0,:,:,0]).save("remove1.png")



