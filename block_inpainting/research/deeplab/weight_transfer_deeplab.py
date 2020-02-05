from __future__ import division
from __future__ import with_statement

import numpy
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import gfile
import os
import utils.imageutils as imageutils

import six
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import train_utils
from deployment import model_deploy

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('input_channels', 5, 'number of input channels')

flags.DEFINE_enum('input_kernel_filler', 'gaussian',
                  ['zeros', 'gaussian'], 'Weight filler for additional input channels not present in checkpoint.')

# Settings for logging.
flags.DEFINE_multi_integer('train_crop_size', [513, 513],
                           'Image crop size [height, width] during training.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

flags.DEFINE_string('source_checkpoint_dir', None,
                    'The source checkpoint in tensorflow format.')

flags.DEFINE_string('source_checkpoint_name', 'model.ckpt',
                    'The source checkpoint name.')

flags.DEFINE_string('output_checkpoint_dir', None,
                    'The output checkpoint in tensorflow format.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during
# training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_integer('num_classes', 21, '')


#def load(sess):
#    with gfile.FastGFile('deeplabv3_pascal_train_aug/frozen_inference_graph.pb', 'rb') as f:
#        graph_def = tf.GraphDef()
#        graph_def.ParseFromString(f.read())
#        sess.graph.as_default()
#        tf.import_graph_def(graph_def, name='')


#def main_frozen():
#    '''Load frozen graph, try to do inference'''
#    data = imageutils.resize(imageutils.read(
#        'deeplab/Oxford.street.london.arp.jpg'), (128, 192))
#    with tf.Session() as sess:
#        load(sess)
#        #[n.name for n in tf.get_default_graph().as_graph_def().node]
#        X = tf.get_default_graph().get_tensor_by_name('SemanticPredictions/begin:0')
#        output = tf.get_default_graph().get_tensor_by_name('SemanticPredictions:0')
#        # what next?


def get_tensor_value(a, sess, verbose=False):
    if isinstance(a, tf.Variable):
        z = a.value().eval(session=sess)
    elif isinstance(a, tf.Tensor):
        z = a.eval(session=sess)
    else:
        raise ValueError
    if verbose:
        print('min, mean, max, std = {}, {}, {}, {}'.format(
            z.min(), z.mean(), z.max(), z.std()))
    return z


def joint_center_slice(a, b):
    '''
    Given numbers a and b, returns a destination and source slice such that the center is taken.

    >>> joint_center_slice(3,5)
    (slice(None, None, None), slice(1, 4, None))
    >>> joint_center_slice(5,3)
    (slice(1, 4, None), slice(None, None, None))
    >>> joint_center_slice(3,3)
    (slice(None, None, None), slice(None, None, None))
    '''
    if a == b:
        s1 = slice(None)
        s2 = slice(None)
    elif a <= b:
        s1 = slice(None)
        s2 = slice((b - a) // 2, (b - a) // 2 + a)
    else:
        s1 = slice((a - b) // 2, (a - b) // 2 + b)
        s2 = slice(None)
    return s1, s2


def joint_begin_slice(a, b):
    '''
    Given numbers a and b, returns a destination and source slice such that the beginning is taken.

    >>> joint_begin_slice(3,5)
    (slice(None, None, None), slice(0, 3, None))
    >>> joint_begin_slice(5,3)
    (slice(0, 3, None), slice(None, None, None))
    >>> joint_begin_slice(3,3)
    (slice(None, None, None), slice(None, None, None))
    '''
    if a == b:
        s1 = slice(None)
        s2 = slice(None)
    elif a <= b:
        s1 = slice(None)
        s2 = slice(0, a)
    else:
        s1 = slice(0, b)
        s2 = slice(None)
    return s1, s2


def joint_end_slice(a, b):
    '''
    Given numbers a and b, returns a destination and source slice such that the ending is taken.

    >>> joint_end_slice(3,5)
    (slice(None, None, None), slice(2, 5, None))
    >>> joint_end_slice(5,3)
    (slice(2, 5, None), slice(None, None, None))
    >>> joint_end_slice(3,3)
    (slice(None, None, None), slice(None, None, None))
    '''
    if a == b:
        s1 = slice(None)
        s2 = slice(None)
    elif a <= b:
        s1 = slice(None)
        s2 = slice(b - a, b)
    else:
        s1 = slice(a - b, a)
        s2 = slice(None)
    return s1, s2


def joint_slice(a, b, gravity):
    if gravity == 0:
        return joint_center_slice(a, b)
    elif gravity == (-1):
        return joint_begin_slice(a, b)
    elif gravity == 1:
        return joint_end_slice(a, b)
    else:
        raise ValueError


def partial_assign_variable_op(dest, src, gravity, strict=True, data=None):
    '''
    dest is a tf.Variable.
    src is a numpy.ndarray.
    gravity governs how a smaller tensor is copied into a larger one. gravity 0 centers the data, -1 packs the data toward the front (ie 0 index), 1 packs the data toward the end. For example: gravity (0,0,-1,-1) will take the spatial center of the kernel and the lower-indexed input and output channels.

    Unassigned values will be set to zero. Optionally, the data parameter can be a mutable numpy.ndarray of the same shape and type as dest. In this case, data is mutated by assigned src values and unassigned values are left unchanged. A reference is held to data, so it must not be mutated until after the op is run.

    Returns an op which assigns data to the destination Variable (eval as sess.run(op)).
    '''
    if strict:
        assert isinstance(dest, tf.Variable)
        assert isinstance(src, numpy.ndarray)
    assert len(dest.shape) == len(src.shape) == len(gravity)
    if data is None:
        data = numpy.zeros(dest.shape, dtype=dest.dtype.as_numpy_dtype)
    else:
        assert data.shape == dest.shape
        assert data.dtype == dest.dtype.as_numpy_dtype
    # for each dim, accumulate the dest and source slice
    r = len(dest.shape)
    slices = []
    for k in range(r):
        slices.append(joint_slice(dest.shape[k], src.shape[k], gravity[k]))
    slices = list(zip(*slices))  # interleaved -> sequential
    data[slices[0]] = src[slices[1]]
    op = dest.assign(data)
    return op


def assign_conv2d_from_checkpoint_fn(variable_name, checkpoint_name, checkpoint_tensor_name):
    '''
    Like slim.assign_from_checkpoint_fn, this returns a function that needs to be run in your session (ie f(sess)). This assignment will allow the conv2d kernel shapes to be different. Only the parts that overlap will be copied and non-overlapping parts will be set to zero. The spatial dimensions will be centered so that a bigger/smaller kernel will crop/expand from the center. The feature dimensions will be packed in the lower indices.
    '''
    # variable_name='xception_65/entry_flow/conv1_1/weights:0'
    # checkpoint_name='deeplabv3_pascal_train_aug/model.ckpt'
    # checkpoint_tensor_name='xception_65/entry_flow/conv1_1/weights'
    model_vars = slim.get_model_variables()
    dest = next(x for x in model_vars if x.name ==
                variable_name)  # tf.Variable
    src = tf.contrib.framework.load_variable(
        checkpoint_name, checkpoint_tensor_name)  # numpy.ndarray

    if FLAGS.input_kernel_filler == 'gaussian':
        # Gaussian init
        n = numpy.array(dest.get_shape().as_list())
        #print(n)
        init_data = numpy.random.randn(numpy.product(n)) * numpy.sqrt(2.0 / numpy.product(n))
        init_data = init_data.reshape(n).astype(dest.dtype.as_numpy_dtype)
        #print (init_data.shape)
    elif FLAGS.input_kernel_filler == 'zeros':
        # Zero init
        init_data = None
    else:
        raise Exception("Unknown filler type: {}".format(FLAGS.input_kernel_filler))

    op = partial_assign_variable_op(dest=dest,
                                    src=src,
                                    gravity=(0, 0, -1, -1),
                                    data=init_data)

    def f(sess):
        sess.run(op)
    return f


def _build_deeplab_inputs(model_inputs, outputs_to_num_classes):
    """Builds a clone of DeepLab.
    MODIFIED FROM train.py-->_build_deeplab.
    The purpose of this function is just to build the model.
    """
    model_options = common.ModelOptions(
        outputs_to_num_classes=outputs_to_num_classes,
        crop_size=FLAGS.train_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)
    outputs_to_scales_to_logits = model.multi_scale_logits(
        model_inputs,
        model_options=model_options,
        image_pyramid=FLAGS.image_pyramid,
        weight_decay=0.01, #weight_decay=FLAGS.weight_decay,
        is_training=True,
        fine_tune_batch_norm=True) #FLAGS.fine_tune_batch_norm)

    return outputs_to_scales_to_logits

def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default() as graph:
        num_channels = FLAGS.input_channels
        num_channels = max(3, num_channels)
        # simulate num_channels channel input
        data = imageutils.resize(imageutils.read(
            'deeplab/Oxford.street.london.arp.jpg'), (128, 192)).reshape(1, 128, 192, 3)
        inputs = tf.to_float(numpy.concatenate([data[...,0:1]]*num_channels, axis=3))

        # Create the global step on the device storing the variables.
        global_step = tf.train.get_or_create_global_step()

        # Define the model and create clones.
        model_fn = _build_deeplab_inputs

        num_classes = FLAGS.num_classes
        model_args = (inputs,
                      {common.OUTPUT_TYPE: num_classes})
        model_fn(*model_args)

        # Soft placement allows placing on CPU ops without GPU implementation.
        session_config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)

        ### Adapted from code by Paul Upchurch. ###
        model_vars = slim.get_model_variables()  # debug
        #print("##2##")
        #print(model_vars)

        if FLAGS.model_variant == 'xception_65':
            input_kernel_name = 'xception_65/entry_flow/conv1_1/weights'
        elif FLAGS.model_variant == 'mobilenet_v2':
            input_kernel_name = 'MobilenetV2/Conv/weights'
        else:
            raise Exception("{} is not supported. Modify the code.".format(FLAGS.model_variant))
        variables_to_restore = slim.get_variables_to_restore(
            exclude=['global_step', input_kernel_name])

        #### Deeplab ####
        checkpoint_dir = FLAGS.source_checkpoint_dir
        checkpoint_name = FLAGS.source_checkpoint_name
        loader = slim.assign_from_checkpoint_fn(
            checkpoint_dir+'/' + checkpoint_name, variables_to_restore, ignore_missing_vars=False)
        ################

        init_op = tf.global_variables_initializer()
        #print ('##3## init_op...')
        #print(init_op)

        saver = tf.train.Saver(tf.global_variables())
        with tf.Session(config=session_config) as sess:
            sess.run(init_op)
            loader(sess)
            f = assign_conv2d_from_checkpoint_fn(input_kernel_name+':0',
                                                checkpoint_dir+'/' + checkpoint_name,
                                                input_kernel_name)

            f(sess)

            print('== Expanded kernel, first output feature ==')
            print(get_tensor_value(model_vars[0], sess).shape)
            print(get_tensor_value(model_vars[0], sess)[:, :, :, 0])
            print('== Original kernel, first output feature ==')
            print(tf.contrib.framework.load_variable(checkpoint_dir+'/' + checkpoint_name,
                                                    input_kernel_name).shape)
            print(tf.contrib.framework.load_variable(checkpoint_dir+'/' + checkpoint_name,
                                                    input_kernel_name)[:, :, :, 0])

            output_dir = FLAGS.output_checkpoint_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            print(' == == ==')
            print('Saving to {}'.format(output_dir))
            saver.save(sess, os.path.join(output_dir, "model.ckpt"))


if __name__ == '__main__':
    flags.mark_flag_as_required('source_checkpoint_dir')
    flags.mark_flag_as_required('output_checkpoint_dir')
    tf.app.run()

