# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
#图像读取库
from PIL import Image
from skimage import transform,io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#矩阵运算库

######
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#####

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
batch_size = 500


def spp_layer(input_, levels=3, name = 'SPP_layer',pool_type = 'max_pool'):

    '''
    Multiple Level SPP layer.
    
    Works for levels=[1, 2, 3, 6].
    '''
    
    shape = input_.get_shape().as_list()
    #print(shape)
    
    with tf.variable_scope(name):

        for l in range(levels):
        #设置池化参数
            l = l + 1
            ksize = [1, np.ceil(shape[1]/ l + 1).astype(np.int32), np.ceil(shape[2] / l + 1).astype(np.int32), 1]
            strides = [1, np.floor(shape[1] / l + 1).astype(np.int32), np.floor(shape[2] / l + 1).astype(np.int32), 1]
            
            if pool_type == 'max_pool':
                pool = tf.nn.max_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                #print(pool)
                pool = tf.reshape(pool,(shape[0],-1),)
                
            else:
                pool = tf.nn.avg_pool(input_, ksize=ksize, strides=strides, padding='SAME')
                pool = tf.reshape(pool,(shape[0],-1))
            print("Pool Level {:}: shape {:}".format(l, pool.get_shape().as_list()))
            if l == 1:
                x_flatten = tf.reshape(pool,(shape[0],-1))
            else:
                x_flatten = tf.concat((x_flatten,pool),axis=1) #四种尺度进行拼接
            print("Pool Level {:}: shape {:}".format(l, x_flatten.get_shape().as_list()))
            # pool_outputs.append(tf.reshape(pool, [tf.shape(pool)[1], -1]))
            

    return x_flatten


def inference(X,n_outputs,training):
	X_reshaped = tf.reshape(X, shape=[batch_size, 75, 50, 3])

	with tf.name_scope('cnn'):
	    conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,strides=conv1_stride, padding=conv1_pad, activation=None, name="conv1")
	    conv1_bn = tf.layers.batch_normalization(conv1,training=training)
	    conv1_bn_act = tf.nn.relu(conv1_bn)
	
	    pool1 = tf.nn.max_pool(conv1_bn_act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
	    conv2 = tf.layers.conv2d(pool1, filters=conv2_fmaps, kernel_size=conv2_ksize,strides=conv2_stride, padding=conv2_pad,activation=None, name="conv2")
	    conv2_bn = tf.layers.batch_normalization(conv2,training=training)
	    conv2_bn_act = tf.nn.relu(conv2_bn)
	
	    spp = spp_layer(conv2_bn_act,4)
	    print(spp.shape)

	    fc1 = tf.layers.dense(spp, n_fc1, activation=None, name="fc1")
	    fc1_bn =  tf.layers.batch_normalization(fc1,training=training)
	    fc1_bn_act = tf.nn.relu(fc1_bn)

	    fc1_drop = tf.layers.dropout(fc1_bn_act, fc1_dropout_rate, training=training)
	    #print(fc1.shape,fc1_bn_act.shape)

	with tf.name_scope("output"):
	    logits = tf.layers.dense(spp, n_outputs, name="output")
	    Y_proba = tf.nn.softmax(logits, name="Y_proba")
	    #print(Y_proba.shape)
	return logits

def losses(logits, labels):
    with tf.name_scope("loss"):
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(y, n_outputs))
        loss = tf.reduce_mean(xentropy) 
    return loss

def trainning(loss):
    optimizer = tf.train.AdamOptimizer()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops): #要先执行完update_ops操作后才能开始学习
        training_op = optimizer.minimize(loss)
    return training_op

def evaluation(logits, labels):
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    return accuracy



