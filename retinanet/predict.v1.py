import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
from sys import path
import cv2
import tensorflow as tf
from skimage import transform
import torch
from torchvision import  models, transforms
import skimage.io
from retinanet.dataloader import  Resizer, collater, Normalizer, UnNormalizer
from retinanet import cnn
def load_model():
    """
        Loading the pre-trained model and parameters.
    """
    global X, yhat
    modelpath = r'../cnn/model/'
    saver = tf.train.import_meta_graph(modelpath + 'image_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint(modelpath))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    yhat = graph.get_tensor_by_name("logit:0")
    print('Successfully load the pre-trained model!')

def predict(txtdata):
    """
        Convert data to Numpy array which has a shape of (-1, 41, 41, 41 3).
        Test a single example.
        Arg:
                txtdata: Array in C.
        Returns:
            Three coordinates of a face normal.
    """
    global X, yhat

    data = np.array(txtdata)
    data = data.reshape(-1, 41, 41, 41, 3)
    output = sess.run(yhat, feed_dict={X: data})  # (-1, 3)
    output = output.reshape(-1, 1)
    ret = output.tolist()
    return ret



def predict(image):
    start = time.time()
    model_path = 'model_final.pt'
    if len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)
    image = image.astype(np.float32)/255.0
    print('time1: {}'.format(time.time()-start))  #0.0005311965942382812
    transform_torch=transforms.Compose([Normalizer(), Resizer()])
    data = transform_torch(image)
    print('time1.5: {}'.format(time.time()-start))  #0.0962982177734375   0.095
    data = collater(data)
    print('time2: {}'.format(time.time()-start))  #0.11830759048461914   0.022

    retinanet = torch.load(model_path)
    print('time3: {}'.format(time.time()-start))  #0.22663378715515137   0.108

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    unnormalize = UnNormalizer()
    print('time4: {}'.format(time.time()-start))  #0.23058390617370605   0.0039
    with torch.no_grad():
        st = time.time()
        if torch.cuda.is_available():
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
        else:
            scores, classification, transformed_anchors = retinanet(data['img'].float())
        print('time5: {}'.format(time.time()-st))  #0.058475494384765625   
        idxs = np.where(scores.cpu()>0.5)
        img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
        img[img<0] = 0
        img[img>255] = 255
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        print('time6: {}'.format(time.time()-st))  #0.06939291954040527   0.0109175
        resu = []
        key1 = "box"
        key2 = "number"
        for j in range(idxs[0].shape[0]):
            resu_dict={}
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            cropped = img[y1:y2, x1:x2]  # 裁剪坐标为[y0:y1, x0:x1]
            cropped = transform.resize(cropped, (75, 50))
            cropped = cropped.astype(np.float32).reshape(-1, 75*50, 3) / 255.0
            print('time7: {}'.format(time.time()-st))  #0.07093954086303711    0.00154663
            with tf.Graph().as_default():
                logit = cnn.inference(cropped, 10, False)
                print(logit)
                logit = tf.nn.softmax(logit)
                X = tf.placeholder(tf.float32, shape=[1, 75*50, 3], name="X")
                saver = tf.train.Saver()
                with tf.Session() as sess:
                    saver.restore(sess, '../cnn/model/image_model')
                    prediction = sess.run(logit, feed_dict={X: cropped})
                    max_index = np.argmax(prediction)
                    print('time8: {}'.format(time.time()-st))  #1.1471683979034424    1.07622846
                    resu_dict.setdefault(key1,[]).append(str(x1))
                    resu_dict.setdefault(key1,[]).append(str(y1))
                    resu_dict.setdefault(key1,[]).append(str(x2))
                    resu_dict.setdefault(key1,[]).append(str(y2))
                    resu_dict.setdefault(key2,[]).append(str(max_index))
            resu.append(resu_dict)
        return(resu)

if __name__ == '__main__':
    img = skimage.io.imread('/home/wangzj/WORK/kaiguan/csv/data/000237.jpg')
    resu = predict(img)
    print(resu)

