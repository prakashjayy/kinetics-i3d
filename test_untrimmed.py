""" Testing on untrimmed videos

Given a video_loc (with images and optical flows), It generates a .csv file with top-n categories
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf

import i3d
from utils import *


_VID_LOC = "vid/Chicago_Holly_B_Kitchen_20141221113134/" ## set location to dense_flow outputs of a video
_SAVE_LOC = "pg_test_videos/" ## out_put where the csv file need to be saved.

_IMAGE_SIZE = 224
_NUM_CLASSES = 400

_SAMPLE_VIDEO_FRAMES = 79

_RESIZE_LEN = 256
_CROP_W = 224
_CROP_H = 224


_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = FLAGS.eval_type
  imagenet_pretrained = FLAGS.imagenet_pretrained

  if eval_type not in ['rgb', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')

  kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  if eval_type in ['rgb', 'joint']:
    # RGB input has 3 channels.
    rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))
    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)
    rgb_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'RGB':
        rgb_variable_map[variable.name.replace(':0', '')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
    flow_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
    with tf.variable_scope('Flow'):
      flow_model = i3d.InceptionI3d(
          _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      flow_logits, _ = flow_model(
          flow_input, is_training=False, dropout_keep_prob=1.0)
    flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

  if eval_type == 'rgb':
    model_logits = rgb_logits
  elif eval_type == 'flow':
    model_logits = flow_logits
  else:
    model_logits = rgb_logits + flow_logits
  model_predictions = tf.nn.softmax(model_logits)

  with tf.Session() as sess:
      if eval_type in ['flow', 'joint']:
          if imagenet_pretrained:
              rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
          else:
              rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
      if eval_type in ['flow', 'joint']:
          if imagenet_pretrained:
              flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
          else:
              flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])

      final_dict=[]
      count=0
      for m, n, rgbs, flows in generate_numpy_files(_VID_LOC, _RESIZE_LEN, _CROP_H, _CROP_W, _SAMPLE_VIDEO_FRAMES):
          kk = {}
          feed_dict = {}
          feed_dict[rgb_input] = rgbs
          feed_dict[flow_input] = flows

          out_logits, out_predictions = sess.run(
            [model_logits, model_predictions],
            feed_dict=feed_dict)

          out_logits = out_logits[0]
          out_predictions = out_predictions[0]
          sorted_indices = np.argsort(out_predictions)[::-1]

          print('Norm of logits: %f' % np.linalg.norm(out_logits))
          print('\nTop classes and probabilities')
          kk["start_frame"] = m
          kk["end_frame"] = n
          for index in sorted_indices[:10]:
              print(out_predictions[index], out_logits[index], kinetics_classes[index])
              kk["class_"+str(index)] = kinetics_classes[index]
              kk["logits_"+str(index)] = out_logits[index]
              kk["prob_score_"+str(index)] = out_predictions[index]
          final_dict.append(kk)

      final_dict = pd.DataFrame(final_dict)
      final_dict.to_save(_SAVE_LOC+_VID_LOC.rsplit("/")[-2]+".csv", index=False)


if __name__ == '__main__':
  tf.app.run(main)
