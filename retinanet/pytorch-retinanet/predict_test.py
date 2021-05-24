import os
import torch
import skimage
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer



device = torch.device('cuda')
transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
                            ])
unnormalize = UnNormalizer()
def predict(img_path,out_name):
    net=torch.load('./model_final.pt')
    net=net.to(device)
    torch.no_grad()
    img=Image.open(img_path)
    img=transform(img).unsqueeze(0)
    scores, classification, transformed_anchors = net(img.cuda().float())
    print(transformed_anchors)
    x1 = int(transformed_anchors[0][0])
    y1 = int(transformed_anchors[0][1])
    x2 = int(transformed_anchors[0][2])
    y2 = int(transformed_anchors[0][3])
    print(x1,y1,x2,y2)
    img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
    img[img<0] = 0
    img[img>255] = 255
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    cropped = img[y1:y2, x1:x2]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite('./result/crop_{}.jpg'.format(out_name),cropped, [int( cv2.IMWRITE_JPEG_QUALITY), 95])

#    _, predicted = torch.max(outputs, 1)
#    print(predicted)
#    print('this picture maybe :',classes[predicted[0]])
if __name__ == '__main__':
    data_dir = '/home/wangzj/WORK/kaiguan/csv/data/'
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        predict(fpath,fname)
