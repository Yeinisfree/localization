import cv2
import numpy as np
import fl_estimator
import os

file_list = os.listdir("/home/cgv/0306_JUNO/dataset/db_Part2/")
len_db = len(file_list)

print(len_db)
print(file_list[0])