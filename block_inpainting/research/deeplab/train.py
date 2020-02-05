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
"""Training script for the DeepLab model.

See model.py for more details and usage.
"""

import six
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import train_utils
from deployment import model_deploy

slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue

flags = tf.app.flags

FLAGS = flags.FLAGS

# Settings for multi-GPUs/multi-replicas training.

flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.')

flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID.')

# Settings for logging.

flags.DEFINE_string('train_logdir', None,
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('validation_interval', 0,
                     'Stop training every N iterations so evaluation loop can be run. **NOTE**: evaluation loop must be called separately and training must be manually restarted. If set to <= 0, then validation_interval is set to training_number_of_steps')

flags.DEFINE_integer('save_summaries_secs', 600,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as '
                     'images to summary.')

# Settings for training strategy.

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 30000,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
flags.DEFINE_integer('train_batch_size', 8,
                     'The number of images in each batch during training.')

flags.DEFINE_integer('batch_iter', 1,
                     'Accumulate gradients over batch_iter batches. num_clones must be 1 and num_replicas must be 1.')

flags.DEFINE_boolean('batch_iter_verbose', False,
                   'If true, print additional information about batch_iteration.')

flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_multi_integer('train_crop_size', [513, 513],
                           'Image crop size [height, width] during training.')

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

# Settings for fine-tuning the network.

flags.DEFINE_string('tf_initial_checkpoint', None,
                    'The initial checkpoint in tensorflow format.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')

flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

flags.DEFINE_boolean('class_balanced_loss', True,
                     'Default True. If True, use dataset.loss_weight to weight class losses. Default weight is 1.0 for every class if dataset.loss_weight is not defined. If FLAGS.class_balanced_loss=False, then weight is 1.0 for every class.')

# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Fine tune the batch norm parameters or not.')

flags.DEFINE_float('min_scale_factor', 0.5,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 2.,
                   'Maximum scale factor for data augmentation.')

flags.DEFINE_float('scale_factor_step_size', 0.25,
                   'Scale factor step size for data augmentation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during
# training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.
flags.DEFINE_string('dataset', 'cityscapes',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')


def _build_deeplab(inputs_queue, outputs_to_num_classes, ignore_label, loss_weight):
    """Builds a clone of DeepLab.

    Args:
      inputs_queue: A prefetch queue for images and labels.
      outputs_to_num_classes: A map from output type to the number of classes.
        For example, for the task of semantic segmentation with 21 semantic
        classes, we would have outputs_to_num_classes['semantic'] = 21.
      ignore_label: Ignore label.
      loss_weight: float or list of floats of length num_classes. Loss weight for each class. Default is 1.0.

    Returns:
      A map of maps from output_type (e.g., semantic prediction) to a
        dictionary of multi-scale logits names to logits. For each output_type,
        the dictionary has keys which correspond to the scales and values which
        correspond to the logits. For example, if `scales` equals [1.0, 1.5],
        then the keys would include 'merged_logits', 'logits_1.00' and
        'logits_1.50'.
    """
    samples = inputs_queue.dequeue()

    # Add name to input and label nodes so we can add to summary.
    samples[common.IMAGE] = tf.identity(
        samples[common.IMAGE], name=common.IMAGE)
    samples[common.LABEL] = tf.identity(
        samples[common.LABEL], name=common.LABEL)

    if FLAGS.input_hints:

        ###
        if 'dynamic_block_hint' in FLAGS.hint_types:
            assert len(FLAGS.hint_types) == 1, 'When using dynamic block hints, do not use other hint types!'
            print("----")
            print("train.py: Block hints with grid {}x{}.".format(FLAGS.dynamic_block_hint_B, FLAGS.dynamic_block_hint_B))
            print("train.py: Drawing blocks with p {}.".format(FLAGS.dynamic_block_hint_p))

            class_hints, hinted = tf.py_func(func=train_utils.generate_class_partial_boundaries_helper(B=FLAGS.dynamic_block_hint_B,
                                                                                                       p=FLAGS.dynamic_block_hint_p),
                                     inp=[samples[common.LABEL]],
                                     Tout=[tf.uint8, tf.bool])
            samples[common.HINT] = class_hints
            samples[common.HINT].set_shape(samples[common.LABEL].get_shape().as_list())
            FLAGS.hint_types = ['class_hint']

        if 'class_hint' in FLAGS.hint_types:
            assert len(FLAGS.hint_types) == 1, 'When using class hints, do not use other hint types!'
            num_classes = outputs_to_num_classes['semantic']
            print('train.py: num semantic classes is {}'.format(num_classes))

            class_hint_channels_list = []
            for label in range(num_classes):
                # Multiply by 255 is to bring into same range as image pixels...,
                # and so feature_extractor mean subtraction will reduce it back to 0,1 range
                class_hint_channel = tf.to_float(tf.equal(samples[common.HINT], label)) * 255
                class_hint_channels_list.append(class_hint_channel)
            class_hint_channels = tf.concat(class_hint_channels_list, axis=-1)
            samples[common.HINT] = class_hint_channels
        ####

        # Get hints and concat to image as input into network
        samples[common.HINT] = tf.identity(
            samples[common.HINT], name=common.HINT)
        model_inputs = tf.concat(
            [samples[common.IMAGE], tf.to_float(samples[common.HINT])], axis=-1)
    else:
        # Just image is input into network
        model_inputs = samples[common.IMAGE]

    model_options = common.ModelOptions(
        outputs_to_num_classes=outputs_to_num_classes,
        crop_size=FLAGS.train_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    print('train.py: FORCE_DROPOUT IS {}'.format(FLAGS.force_dropout))
    if FLAGS.force_dropout:
        print('train.py: FORCE_DROPOUT keep prob {}'.format(FLAGS.keep_prob))
        print('train.py: FORCE_DROPOUT_ONLY_BRANCH IS {}'.format(FLAGS.force_dropout_only_branch))

    outputs_to_scales_to_logits = model.multi_scale_logits(
        model_inputs,
        model_options=model_options,
        image_pyramid=FLAGS.image_pyramid,
        weight_decay=FLAGS.weight_decay,
        is_training=True,
        fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
        force_dropout=FLAGS.force_dropout,
        force_dropout_only_branch=FLAGS.force_dropout_only_branch,
        keep_prob=FLAGS.keep_prob)

    # Add name to graph node so we can add to summary.
    output_type_dict = outputs_to_scales_to_logits[common.OUTPUT_TYPE]
    output_type_dict[model.get_merged_logits_scope()] = tf.identity(
        output_type_dict[model.get_merged_logits_scope()],
        name=common.OUTPUT_TYPE)


    for output, num_classes in six.iteritems(outputs_to_num_classes):

        print('OUTPUTS: {}'.format(output))
        train_utils.add_softmax_cross_entropy_loss_for_each_scale(
            outputs_to_scales_to_logits[output],
            samples[common.LABEL],
            num_classes,
            ignore_label,
            loss_weight=loss_weight,
            upsample_logits=FLAGS.upsample_logits,
            scope=output,
            )


    return outputs_to_scales_to_logits


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.batch_iter < 1:
        FLAGS.batch_iter = 1
    if FLAGS.batch_iter != 1:
        if not (FLAGS.num_clones == 1 and FLAGS.num_replicas == 1):
            raise NotImplementedError("train.py: **NOTE** -- train_utils.train_step_custom may not work with parallel GPUs / clones > 1! Be sure you are only using one GPU.")

    print('\ntrain.py: Accumulating gradients over {} iterations\n'.format(FLAGS.batch_iter))

    # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
    config = model_deploy.DeploymentConfig(
        num_clones=FLAGS.num_clones,
        clone_on_cpu=FLAGS.clone_on_cpu,
        replica_id=FLAGS.task,
        num_replicas=FLAGS.num_replicas,
        num_ps_tasks=FLAGS.num_ps_tasks)

    # Split the batch across GPUs.
    assert FLAGS.train_batch_size % config.num_clones == 0, (
        'Training batch size not divisble by number of clones (GPUs).')

    clone_batch_size = FLAGS.train_batch_size // config.num_clones

    # Get dataset-dependent information.
    dataset = segmentation_dataset.get_dataset(
        FLAGS.dataset, FLAGS.train_split,
        dataset_dir=FLAGS.dataset_dir,
        )

    tf.gfile.MakeDirs(FLAGS.train_logdir)
    tf.logging.info('Training on %s set', FLAGS.train_split)

    with tf.Graph().as_default() as graph:
        with tf.device(config.inputs_device()):
            samples = input_generator.get(
                dataset,
                FLAGS.train_crop_size,
                clone_batch_size,
                min_resize_value=FLAGS.min_resize_value,
                max_resize_value=FLAGS.max_resize_value,
                resize_factor=FLAGS.resize_factor,
                min_scale_factor=FLAGS.min_scale_factor,
                max_scale_factor=FLAGS.max_scale_factor,
                scale_factor_step_size=FLAGS.scale_factor_step_size,
                dataset_split=FLAGS.train_split,
                is_training=True,
                model_variant=FLAGS.model_variant)
            inputs_queue = prefetch_queue.prefetch_queue(
                samples, capacity=128 * config.num_clones)

        # Create the global step on the device storing the variables.
        with tf.device(config.variables_device()):
            global_step = tf.train.get_or_create_global_step()

            # Define the model and create clones.
            model_fn = _build_deeplab
            if FLAGS.class_balanced_loss:
                print('train.py: class_balanced_loss=True. Reading loss weights from segmentation_dataset.py')
            else:
                print('train.py: class_balanced_loss=False. Setting loss weights to 1.0 for every class.')
                dataset.loss_weight = 1.0

            #_build_deeplab has model args:
            #(inputs_queue, outputs_to_num_classes, ignore_label, loss_weight):

            outputs_to_num_classes = {common.OUTPUT_TYPE: dataset.num_classes}

            model_args = (inputs_queue,\
                          outputs_to_num_classes,
                          dataset.ignore_label, dataset.loss_weight)
            clones = model_deploy.create_clones(
                config, model_fn, args=model_args)

            # Gather update_ops from the first clone. These contain, for example,
            # the updates for the batch_norm variables created by model_fn.
            first_clone_scope = config.clone_scope(0)
            update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for model variables.
        for model_var in slim.get_model_variables():
            summaries.add(tf.summary.histogram(model_var.op.name, model_var))

        # Add summaries for images, labels, semantic predictions
        if FLAGS.save_summaries_images:
            summary_image = graph.get_tensor_by_name(
                ('%s/%s:0' % (first_clone_scope, common.IMAGE)).strip('/'))
            summaries.add(
                tf.summary.image('samples/%s' % common.IMAGE, summary_image))

            first_clone_label = graph.get_tensor_by_name(
                ('%s/%s:0' % (first_clone_scope, common.LABEL)).strip('/'))
            # Scale up summary image pixel values for better visualization.
            pixel_scaling = max(1, 255 // dataset.num_classes)
            summary_label = tf.cast(
                first_clone_label * pixel_scaling, tf.uint8)
            summaries.add(
                tf.summary.image('samples/%s' % common.LABEL, summary_label))

            first_clone_output = graph.get_tensor_by_name(
                ('%s/%s:0' % (first_clone_scope, common.OUTPUT_TYPE)).strip('/'))
            predictions = tf.expand_dims(tf.argmax(first_clone_output, 3), -1)

            summary_predictions = tf.cast(
                predictions * pixel_scaling, tf.uint8)
            summaries.add(
                tf.summary.image(
                    'samples/%s' % common.OUTPUT_TYPE, summary_predictions))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Build the optimizer based on the device specification.
        with tf.device(config.optimizer_device()):
            learning_rate = train_utils.get_model_learning_rate(
                FLAGS.learning_policy, FLAGS.base_learning_rate,
                FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
                FLAGS.training_number_of_steps, FLAGS.learning_power,
                FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
            optimizer = tf.train.MomentumOptimizer(
                learning_rate, FLAGS.momentum)
            summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        startup_delay_steps = FLAGS.task * FLAGS.startup_delay_steps
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        with tf.device(config.variables_device()):

            total_loss, grads_and_vars = model_deploy.optimize_clones(
                clones, optimizer)
            total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')

            # Modify the gradients for biases and last layer variables.
            last_layers = model.get_extra_layer_scopes(
                FLAGS.last_layers_contain_logits_only)
            grad_mult = train_utils.get_model_gradient_multipliers(
                last_layers, FLAGS.last_layer_gradient_multiplier)
            if grad_mult:
                grads_and_vars = slim.learning.multiply_gradients(
                    grads_and_vars, grad_mult)

            if FLAGS.batch_iter <= 1:
                FLAGS.batch_iter = 0
                summaries.add(tf.summary.scalar('total_loss', total_loss))
                grad_updates = optimizer.apply_gradients(
                    grads_and_vars, global_step=global_step)
                update_ops.append(grad_updates)
                update_op = tf.group(*update_ops)
                with tf.control_dependencies([update_op]):
                    train_tensor = tf.identity(total_loss, name='train_op')
                accum_tensor = None
            else:

                ############ Accumulate grads_and_vars op. ####################
                accum_update_ops = list(update_ops) #.copy()
                # Create (grad, var) list to accumulate gradients in. Inititalize to 0.
                accum_grads_and_vars = [(tf.Variable(tf.zeros_like(gv[0]), trainable=False, name=gv[0].name.strip(":0")+"_accum"), gv[1] ) for gv in grads_and_vars]
                assert len(accum_grads_and_vars) == len(grads_and_vars)

                total_loss_accum = tf.Variable(0.0, dtype=tf.float32, trainable=False)
                accum_loss_update_op = [total_loss_accum.assign_add(total_loss)]
                accum_update_ops.append(accum_loss_update_op)

                ## Accumulate gradients: accum_grad[i] += (grad[i] / FLAGS.batch_iter)  # scaled gradients.
                accum_ops = [accum_grads_and_vars[i][0].assign_add(tf.div(gv[0], 1.0 * FLAGS.batch_iter)) for i, gv in enumerate(grads_and_vars)]
                accum_update_ops.append(accum_ops)

                accum_update_op = tf.group(*accum_update_ops)
                with tf.control_dependencies([accum_update_op]):
                    accum_print_ops = []
                    if FLAGS.batch_iter_verbose:
                        accum_print_ops.extend([
                                                        tf.Print(tf.constant(0), [tf.add(global_step, 1)], message='train.py: accumulating gradients for step: '),
                            #tf.Print(total_loss, [total_loss], message='    step total_loss: ')
                            #tf.Print(tf.constant(0), [accum_grads_and_vars[0][0]], message='    '),
                                        ])
                    accum_update_ops.append(accum_print_ops)
                    with tf.control_dependencies([tf.group(*accum_print_ops)]):
                        accum_tensor = tf.identity(total_loss_accum, name='accum_op')

                ##################### Train op (apply [accumulated] grads and vars) ###############################
                train_update_ops = list(update_ops) #.copy()
                ## Create gradient update op.
                # Apply gradients from accumulated gradients
                grad_updates = optimizer.apply_gradients(
                    accum_grads_and_vars, global_step=global_step)
                train_update_ops.append(grad_updates)

                grad_print_ops = []
                if FLAGS.batch_iter_verbose:
                    grad_print_ops.extend([
        #                tf.Print(tf.constant(0), [grads_and_vars[0][0], grads_and_vars[0][1]], message='---grads[0] and vars[0]---------\n'),
                        #tf.Print(tf.constant(0), [], message=grads_and_vars[0][1].name),
                        tf.Print(tf.constant(0), [accum_grads_and_vars[0][0]], message='GRADS  BEFORE ZERO: ')
                    ])
                train_update_ops.append(grad_print_ops)

                total_loss_accum_average = tf.div(total_loss_accum, FLAGS.batch_iter)
                summaries.add(tf.summary.scalar('total_loss', total_loss_accum_average))

                train_update_op = tf.group(*train_update_ops)
                with tf.control_dependencies([train_update_op]):
                    zero_ops = []

                    zero_accum_ops = [agv[0].assign(tf.zeros_like(agv[0])) for agv in accum_grads_and_vars]
                    zero_ops.append(zero_accum_ops)

                    zero_accum_total_loss_op = [total_loss_accum.assign(0)]
                    zero_ops.append(zero_accum_total_loss_op)

                    zero_op = tf.group(*zero_ops)
                    with tf.control_dependencies([zero_op]):
                        grad_print_ops = []
                        if FLAGS.batch_iter_verbose:
                            grad_print_ops.extend([
                                #tf.Print(tf.constant(0), [accum_grads_and_vars[0][0]], message='GRADS AFTER ZERO ')
                            ])
                        with tf.control_dependencies([tf.group(*grad_print_ops)]):
                            train_tensor = tf.identity(total_loss_accum_average, name='train_op')

        # Add the summaries from the first clone. These contain the summaries
        # created by model_fn and either optimize_clones() or
        # _gather_clone_loss().
        summaries |= set(
            tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries))

        # Soft placement allows placing on CPU ops without GPU implementation.
        session_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)

        session_config.gpu_options.allow_growth = True

        #train_step_exit = train_utils.train_step_exit
        train_step_custom = train_utils.train_step_custom
        if FLAGS.validation_interval <= 0:
            FLAGS.validation_interval = FLAGS.training_number_of_steps
        else:
            print("*** Validation interval: {} ***".format(FLAGS.validation_interval))

        # Start the training.
        slim.learning.train(
            train_tensor,
            logdir=FLAGS.train_logdir,
            train_step_fn=train_step_custom(VALIDATION_N=FLAGS.validation_interval,
                                          ACCUM_OP=accum_tensor,
                                          ACCUM_STEPS=FLAGS.batch_iter),
            log_every_n_steps=FLAGS.log_steps,
            master=FLAGS.master,
            number_of_steps=FLAGS.training_number_of_steps,
            is_chief=(FLAGS.task == 0),
            session_config=session_config,
            startup_delay_steps=startup_delay_steps,
            init_fn=train_utils.get_model_init_fn(
                FLAGS.train_logdir,
                FLAGS.tf_initial_checkpoint,
                FLAGS.initialize_last_layer,
                last_layers,
                ignore_missing_vars=True),
            summary_op=summary_op,
            save_summaries_secs=FLAGS.save_summaries_secs,
            save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
    flags.mark_flag_as_required('train_logdir')
    flags.mark_flag_as_required('tf_initial_checkpoint')
    flags.mark_flag_as_required('dataset_dir')


    tf.app.run()
