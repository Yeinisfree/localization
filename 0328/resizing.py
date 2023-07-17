import cv2
import os

path = '/home/cgv/0328/part3_AF_rotated_0329'
save = '/home/cgv/0328/part3_AF_rotated_0329'
list = os.listdir(path)

resolution = (640, 480)

imgs = []
for i in list:
    img = cv2.imread(path + '/' + i)
    imgResized = cv2.resize(img, resolution)
    cv2.imwrite(save + '/' + i, imgResized)