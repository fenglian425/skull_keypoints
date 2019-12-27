import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

img_dir = 'data/origin/img'
label_dir = 'data/origin/label'

b = np.zeros((256,256), dtype=np.uint8)
b[:10,:10]=255
cv2.imshow('100', b)

cv2.waitKey()
ratios = []
for img_file in os.listdir(img_dir):
    label_file = img_file.replace('jpg', 'txt')

    coords = np.loadtxt(os.path.join(label_dir, label_file), dtype=np.int, delimiter=',')

    img = cv2.imread(os.path.join(img_dir, img_file))
    height, width = img.shape[:2]
    ratios.append(width / height)
    # print(img.shape)
    # for i in range(coords.shape[0]):
    #     point = coords[i]
    #     point = (int(point[0]), int(point[1]))
    #     print(point)
    #     cv2.circle(img, point, 3, (0, 0, 255))
    #
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # print(img, label_file)

plt.hist(ratios)
plt.show()