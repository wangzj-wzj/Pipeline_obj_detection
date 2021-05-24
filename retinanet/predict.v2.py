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

def load_model(sess):
    """
        Loading the pre-trained model and parameters.
    """
    global X, yhat
    modelpath = r'/home/wangzj/WORK/number/tf-cnn-0330/model/'
   # modelpath = r'/home/xuyiwen/work/llh/cnn/model/'
    saver = tf.train.import_meta_graph(modelpath + '/image_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint(modelpath))
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("inputs/X:0")
    yhat = graph.get_tensor_by_name("output/Y_proba:0")

def predict_cnn(data,sess):
    """
    """
    global X, yhat

    output = sess.run(yhat, feed_dict={X: data})
    return output



def predict(image):
    start = time.time()
    model_path = 'model_final.pt'
    ori_imgshape = image.shape
    if len(image.shape) == 2:
            image = skimage.color.gray2rgb(image)
    image = image.astype(np.float32)/255.0
   # print('time1: {}'.format(time.time()-start))  
    start = time.time()
    transform_torch=transforms.Compose([Normalizer(), Resizer()])
    data = transform_torch(image)
   # print('time1.5: {}'.format(time.time()-start))
    start = time.time()
    data = collater(data)
   # print('time2: {}'.format(time.time()-start))  
    start = time.time()
    retinanet = torch.load(model_path)
   # print('time3: {}'.format(time.time()-start))  
    start = time.time()
#    use_gpu = True

#    if use_gpu:
#        if torch.cuda.is_available():
#            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    unnormalize = UnNormalizer()
    scale = data['scale']
   # print('time4: {}'.format(time.time()-start))  
    with torch.no_grad():
        st = time.time()
        if torch.cuda.is_available():
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
        else:
            scores, classification, transformed_anchors = retinanet(data['img'].float())
       # print('time5: {}'.format(time.time()-st))  
        st = time.time()
        idxs = np.where(scores.cpu()>0.5)
        img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
        img[img<0] = 0
        img[img>255] = 255
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        after_imgshape = img.shape
       # print('after_img.shape{}'.format(img.shape))
       # print('time6: {}'.format(time.time()-st))  
        st = time.time()
        resu = []
        key1 = "box"
        key2 = "number"
        sess = tf.Session()
        X = None # input
        yhat = None # output
        load_model(sess) 
       # print('time7: {}'.format(time.time()-st))  

        for j in range(idxs[0].shape[0]):
            st = time.time()
            resu_dict={}
            bbox = transformed_anchors[idxs[0][j], :]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
#            print(x1,y1,x2,y2)
            cropped = img[y1:y2, x1:x2]  # 裁剪坐标为[y0:y1, x0:x1]
            cropped = transform.resize(cropped, (75, 50))
            cropped = cropped.astype(np.float32).reshape(-1, 75*50, 3) / 255.0
            prediction = predict_cnn(cropped,sess)
            max_index = np.argmax(prediction)
        #    print('time8: {}'.format(time.time()-st)) 
            st = time.time()
            x1 = int(np.divide(x1, scale))
            y1 = int(np.divide(y1, scale))
            x2 = int(np.divide(x2, scale))
            y2 = int(np.divide(y2, scale))
            box = [x1,y1,x2,y2]
            resu_dict.setdefault(key1,box)
            resu_dict.setdefault(key2,str(max_index))
            resu.append(resu_dict)
         #   print('time9: {}'.format(time.time()-st))
        return(resu)
if __name__ == '__main__':


#    alll=os.walk('/home/wangzj/WORK/number/tf-cnn-0330/testdata/')
#    for path,dirr,filelist in alll:
#       for filename in filelist:
#          if filename.endswith('jpg'):
#               img= skimage.io.imread(path+filename)
#               print('original_img.shape{}'.format(img.shape))
#               resu = predict(img)
#               print(resu)


    img = skimage.io.imread('/home/wangzj/WORK/number/tf-cnn-0330/traindata/7_52.jpg')
    print('original_img.shape{}'.format(img.shape))
    resu = predict(img)
    print(resu)
