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
from skimage import transform,io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import  CSVDataset, collater, Resizer, Augmenter, \
    UnNormalizer, Normalizer
from retinanet import cnn



def predict(img_path):

    csv_classes = sys.path[0]+'/csv/class.csv'
    model_path = 'model_final.pt'
   
    dataset_val = CSVDataset(train_file=img_path, class_list=csv_classes, transform=transforms.Compose([Normalizer(), Resizer()]))
    for idx, data in enumerate(dataset_val):
        print(data)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater)

    retinanet = torch.load(model_path)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.eval()

    unnormalize = UnNormalizer()
    
    for idx, data in enumerate(dataloader_val):
        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(data['img'].float())
            print(data)
            print(type(data['img']))
            idxs = np.where(scores.cpu()>0.5)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img<0] = 0
            img[img>255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
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
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                cropped = img[y1:y2, x1:x2]  # 裁剪坐标为[y0:y1, x0:x1]
#                print(cropped.shape)
                cropped = transform.resize(cropped, (75, 50))
                cropped = cropped.astype(np.float32).reshape(-1, 75*50, 3) / 255.0
#                print(cropped)
                with tf.Graph().as_default():
                    logit = cnn.inference(cropped, 10, False)
                    logit = tf.nn.softmax(logit)
                    X = tf.placeholder(tf.float32, shape=[1, 75*50, 3], name="X")

                    saver = tf.train.Saver()
                    with tf.Session() as sess:
                        saver.restore(sess, '../cnn/model/image_model')
                        prediction = sess.run(logit, feed_dict={X: cropped})
                        max_index = np.argmax(prediction)
                        print('Elapsed time: {}'.format(time.time()-st))
#                        print(max_index)
                        resu_dict.setdefault(key1,[]).append(str(x1))
                        resu_dict.setdefault(key1,[]).append(str(y1))
                        resu_dict.setdefault(key1,[]).append(str(x2))
                        resu_dict.setdefault(key1,[]).append(str(y2))
                        resu_dict.setdefault(key2,[]).append(str(max_index))
                resu.append(resu_dict)
            return(resu)



#                cv2.imwrite('{}/crop/crop_{}.jpg'.format(outpath,img_num),cropped, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
#                draw_caption(img, (x1, y1, x2, y2), label_name)
#                print(x1,y1,x2,y2)
#                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
#                print(label_name)
#            cv2.imwrite('{}/photo/photo_{}.jpg'.format(outpath,img_num),img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])

if __name__ == '__main__':
#    img = skimage.io.imread('/home/wangzj/WORK/kaiguan/csv/predict.csv') 
#    resu = predict(img)
    resu = predict('/home/wangzj/WORK/kaiguan/csv/predict.csv')
    print(resu)

