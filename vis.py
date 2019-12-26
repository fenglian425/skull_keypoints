import cv2
import os
import numpy as np


img_dir = 'data/orignial/img'
label_dir = 'data/orignial/label'

for img_file in os.listdir(img_dir):
    label_file = img_file.replace('jpg', 'txt')

    coords = np.loadtxt(os.path.join(label_dir, label_file), dtype=np.int, delimiter=',')

    img = cv2.imread(os.path.join(img_dir, img_file))
    print(img.shape)
    for i in range(coords.shape[0]):
        point = coords[i]
        point = (int(point[0]), int(point[1]))
        print(point)
        cv2.circle(img, point, 3, (0, 0, 255))

    cv2.imshow('img', img)
    cv2.waitKey()
    # print(img, label_file)
