""" Training the RGB model on imagenet_pretrained weights using UCF-101 Dataset
"""

import numpy as np
import tensorflow as tf
import sonnet as snt
import random
from sklearn.metrics import accuracy_score
import glob

import i3d
from utils import *

_SAMPLE_VIDEO_FRAMES = 64
_IMAGE_SIZE = 224
_NUM_CLASSES = 101
_EPOCHS = 10
_BATCH_SIZE = 4

_FILE_LOC_TRAIN = glob.glob("data/train/*.npy")
print("[Total Files: {}]".format(len(_FILE_LOC_TRAIN)))

_MEAN_DATA = np.load("data/mean_data__ucf.npy")[np.newaxis, :, :, :, :]
TRAINING = True
print("Mean_Data: {}".format(_MEAN_DATA.shape))

rgb_input = tf.placeholder(
    tf.float32,
    shape=(None, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

y_true = tf.placeholder(
    tf.float32,
    shape=(None, _NUM_CLASSES))

with tf.variable_scope('RGB'):
  rgb_model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')
  rgb_net, _ = rgb_model( rgb_input, is_training=False, dropout_keep_prob=1.0)
  end_point = 'Logits'
  with tf.variable_scope(end_point):
    rgb_net = tf.nn.avg_pool3d(rgb_net, ksize=[1, 2, 7, 7, 1],
                           strides=[1, 1, 1, 1, 1], padding=snt.VALID)
    if TRAINING:
        rgb_net = tf.nn.dropout(rgb_net, 0.7)
    logits = i3d.Unit3D(output_channels=_NUM_CLASSES,
                    kernel_shape=[1, 1, 1],
                    activation_fn=None,
                    use_batch_norm=False,
                    use_bias=True,
                    name='Conv3d_0c_1x1')(rgb_net, is_training=True)

    logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
    averaged_logits = tf.reduce_mean(logits, axis=1)

  # predictions = tf.nn.softmax(averaged_logits)


rgb_variable_map = {}

for variable in tf.global_variables():
    if variable.name.split("/")[-4] == "Logits": continue
    if variable.name.split('/')[0] == 'RGB':
        rgb_variable_map[variable.name.replace(':0', '')] = variable

#print(rgb_variable_map)
rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

model_logits = averaged_logits
model_predictions = tf.nn.softmax(model_logits)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model_logits,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.MomentumOptimizer(learning_rate=1e-4, momentum=0.9).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    rgb_saver.restore(sess, 'data/checkpoints/rgb_imagenet/model.ckpt')
    print("Model Restored")

    for i in range(_EPOCHS):
        TRAINING = True
        random.shuffle(_FILE_LOC_TRAIN)
        file_loc = _FILE_LOC_TRAIN[0:int(0.2*len(_FILE_LOC_TRAIN))]
        batches = int(len(file_loc)/_BATCH_SIZE)
        for j in tqdm(range(batches)):
            f = file_loc[j * _BATCH_SIZE: (j+1)*_BATCH_SIZE]
            x_train, y_train = npy_reader(f, _MEAN_DATA)
            y_train = one_hot(y_train)

            _, out_predictions = sess.run([optimizer, model_predictions],
                                           feed_dict={rgb_input: x_train, y_true: y_train})
        print("Epoch {} Completed".format(i))
        Actual = []
        predicted = []
        loss = []
        for k, m in tqdm(enumerate(file_loc)):
            TRAINING=False
            x_valid, y_valid = npy_reader_valid(m, _MEAN_DATA)
            y_valid = one_hot([y_valid])
            p_loss, out_predictions = sess.run([cost, model_predictions],
                                                feed_dict={rgb_input: x_valid, y_true: y_valid})
            predicted.append(np.argmax(out_predictions))
            Actual.append(np.argmax(y_valid))
            loss.append(p_loss)
            acc_s = accuracy_score(Actual, predicted)
        #   model_final.save_weights("models/model_2class/model_{}.weights".format(i))
        #   print("Model Saved")
        print("Validation Accuracy: {}, Validation Loss:{}".format(acc_s, sum(loss)/k))
