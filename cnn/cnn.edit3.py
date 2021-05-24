#coding=utf-8

import os
import time
#图像读取库
from PIL import Image
from skimage import transform,io
import matplotlib.pyplot as plt  
import matplotlib.image as mpimg 
import cv2
#矩阵运算库
import numpy as np
import tensorflow as tf

######
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#####

# 数据文件夹
data_dir = "./traindata/"
test_dir = './testdata/'
# 模型文件路径
model_path = "model/image_model"


# 从文件夹读取图片和标签到numpy数组中
# 标签信息在文件名中，例如1_40.jpg表示该图片的标签为1
height = 75
width = 50
channels = 3
n_inputs = height * width

conv1_fmaps = 20
conv1_ksize = 5
conv1_stride = 2
conv1_pad = "SAME"

conv2_fmaps = 40
conv2_ksize = 4
conv2_stride = 2
conv2_pad = "SAME"

pool2_dropout_rate =0 

n_fc1 = 600
fc1_dropout_rate = 0.5
n_outputs = 10
#learning_rate = 0.001

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs, channels], name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    training = tf.placeholder_with_default(False, shape=[], name='training')


with tf.name_scope('cnn'):
    conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad,
                         activation=None, name="conv1")
    conv1_bn = tf.layers.batch_normalization(conv1,training=training)
    conv1_bn_act = tf.nn.relu(conv1_bn)

    pool1 = tf.nn.max_pool(conv1_bn_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv2 = tf.layers.conv2d(pool1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=None, name="conv2")
    conv2_bn = tf.layers.batch_normalization(conv2,training=training)
    conv2_bn_act = tf.nn.relu(conv2_bn)

    pool2 = tf.nn.max_pool(conv2_bn_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    pool2_flat = tf.layers.flatten(pool2)
    pool2_flat_drop = tf.layers.dropout(pool2_flat, pool2_dropout_rate, training=training)
    fc1 = tf.layers.dense(pool2_flat_drop, n_fc1, activation=None
            , name="fc1")
    fc1_bn =  tf.layers.batch_normalization(fc1,training=training)
    fc1_bn_act = tf.nn.relu(fc1_bn)

    fc1_drop = tf.layers.dropout(fc1_bn_act, fc1_dropout_rate, training=training)

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1_drop, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")


with tf.name_scope("train"):
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(y, n_outputs))
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops): #要先执行完update_ops操作后才能开始学习
    training_op = optimizer.minimize(loss)


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

def read_data(data_dir):
    datas = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = mpimg.imread(fpath)[:, :, :channels]
#        print(image)
        image = transform.resize(image, (height, width))
        datas.append(image)
        print('reading the images:%s' % fpath)

    datas = np.array(datas)
    return fpaths, datas

fpaths, X_test = read_data(test_dir)
X_test = X_test.astype(np.float32).reshape(-1, height*width, 3) / 255.0

with tf.Session() as sess:
    print("测试模式")
    saver.restore(sess, model_path)
    print("从{}载入模型".format(model_path))
    # label和名称的对照关系
    label_name_dict = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9"
    }
    predicted_labels_val = sess.run(Y_proba, feed_dict={X: X_test})
    for fpath, predicted_label in zip(fpaths, predicted_labels_val):
        predicted_label_name = label_name_dict[np.argmax(predicted_label)]
        print("{}\t => {}".format(fpath, predicted_label_name))

