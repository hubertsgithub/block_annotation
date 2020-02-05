import numpy as np
import PIL.Image
import glob
import os
import tqdm
from collections import defaultdict


gt_labels = sorted(glob.glob("/PATH/TO/GT/LABELS"))
pred_labels = sorted(glob.glob("/PATH/TO/PRED/LABELS"))


########## Tensorflow mIOU #######
#http://ronny.rest/blog/post_2017_09_11_tf_metrics/
import tensorflow as tf
graph = tf.Graph()
with graph.as_default():
    tf_label = tf.placeholder(dtype=tf.int32, shape=[None])
    tf_prediction = tf.placeholder(dtype=tf.int32, shape=[None])

    weights = tf.to_float(tf.not_equal(tf_label, 0))

    miou, miou_update = tf.metrics.mean_iou(labels=tf_label,
                                            predictions=tf_prediction,
                                            num_classes=134,
                                            weights=weights,
                                            name='tf_miou')

    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope='tf_miou')
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)
with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(running_vars_initializer)

    for gt_label, pred_label in tqdm.tqdm(zip(gt_labels, pred_labels)[::1]):
        gt_label = PIL.Image.open(gt_label)
        gt_label = np.array(gt_label)
        pred_label = PIL.Image.open(pred_label)
        pred_label = np.array(pred_label)

        gt_label = gt_label.flatten()
        pred_label = pred_label.flatten()

        feed_dict={tf_label: gt_label, tf_prediction: pred_label}
        sess.run(miou_update, feed_dict=feed_dict)
    score = sess.run(miou)
    print(score)


########### Per Class mIOU ########
INT = defaultdict(int)
UNI = defaultdict(int)
for gt_label, pred_label in tqdm.tqdm(zip(gt_labels, pred_labels)[::1]):
    gt_label = PIL.Image.open(gt_label)#.resize((513,513), PIL.Image.NEAREST)
    gt_label = np.array(gt_label)

    pred_label = PIL.Image.open(pred_label)#.resize((513,513), PIL.Image.NEAREST)
    pred_label = np.array(pred_label)

    ## Do not mask out to match tf.mIOU computation.
    # Mask out void regions (label 0)
    pred_label[gt_label==0] = 0

    gt_label = gt_label.flatten()
    pred_label = pred_label.flatten()


    for cl in (set(np.unique(gt_label)) | set(np.unique(pred_label))):
        if cl == 0:
           continue
        gt_cl = (gt_label == cl)
        pred_cl = (pred_label == cl)

        INT[cl] += np.sum(np.logical_and(gt_cl, pred_cl))
        UNI[cl] += np.sum(np.logical_or(gt_cl, pred_cl))



#iou = {k: float(TP[k]) / (TP[k] + FN[k] + FP[k]) for k in TP.keys()}
iou = {k: float(INT[k]) / UNI[k] for k in INT.keys()}
miou = np.mean([iou[k] for k in iou.keys()])
print(miou)


###### Per-Image mIOU #######
# Assign maj. GT to simulate class-agnostic mIOU
# This is meant to approximate per-instance mIOU to compare against full-image DEXTR.
from collections import Counter

iou = []
for gt_label, pred_label in tqdm.tqdm(zip(gt_labels, pred_labels)[::1]):
    gt_label = PIL.Image.open(gt_label).resize((513,513), PIL.Image.NEAREST)
    gt_label = np.array(gt_label)

    pred_label = PIL.Image.open(pred_label).resize((513,513), PIL.Image.NEAREST)
    pred_label = np.array(pred_label)

    # Mask out void regions (label 0)
    pred_label[gt_label==0] = 0

    gt_label = gt_label.flatten()
    pred_label = pred_label.flatten()
    new_pred_label = np.zeros_like(pred_label)

    for cl in np.unique(pred_label):
        if cl == 0:
            continue
        new_pred_label[pred_label==cl] = Counter(gt_label[pred_label==cl]).most_common()[0][0]
    pred_label = new_pred_label

    INT = defaultdict(int)
    UNI = defaultdict(int)
    for cl in (set(np.unique(gt_label)) | set(np.unique(pred_label))):
        if cl == 0:
            continue
        gt_cl = (gt_label == cl)
        pred_cl = (pred_label == cl)
        INT[cl] += np.sum(np.logical_and(gt_cl, pred_cl))
        UNI[cl] += np.sum(np.logical_or(gt_cl, pred_cl))

    ious = [float(INT[k]) / UNI[k] for k in INT.keys()]
    iou.extend(ious)

miou = np.mean(iou)
print(miou)



