import cv2
import skimage.io

img = skimage.io.imread('/home/wangzj/WORK/kaiguan/csv/data/001411.jpg')
x1=180
y1=79
x2=241
y2=155
cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
cv2.rectangle(img, (102, 79), (162, 156), color=(0, 0, 255), thickness=2)
cv2.rectangle(img, (50, 79), (109, 157), color=(0, 0, 255), thickness=2)
cv2.rectangle(img, (233, 76), (296, 157), color=(0, 0, 255), thickness=2)
cv2.imwrite('./photo_testresu.jpg',img, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
#cv2.imshow('img',img)
#cv2.waitKey(0)
