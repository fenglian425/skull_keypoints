import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import  COCO
root = 'data/coco'
img_dir = 'data/coco/images'
mode ='train'
anno_file = os.path.join(root, 'annotations', f'instances_{mode}.json')
coco = COCO(anno_file)
img_ids = coco.catToImgs[0]
img_info = coco.loadImgs(img_ids)

img_info = img_info[34]
print(img_info)
im = cv2.imread(os.path.join(img_dir, img_info['file_name']))

anno_ids = coco.getAnnIds(img_info['id'])
print('a',coco.anns)

ratios = [value['bbox'] for key,value in coco.anns.items()]
ratios = [width/height for _,_,width, height in ratios ]
plt.hist(ratios)
plt.show()
print(ratios)
anns = coco.loadAnns(anno_ids)

print(anno_ids)
print(anns)
exit()
#
# plt.imshow(im)
# plt.axis('off')
# coco.showAnns(anns)
# plt.show()
# exit()


img_dir = 'data/origin/img'
label_dir = 'data/origin/label'

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

b = np.zeros((256,256), dtype=np.uint8)
b[:10,:10]=255
cv2.imshow('100', b)

cv2.waitKey()
ratios = []