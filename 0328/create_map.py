import cv2
import numpy as np
import math
import os
import natsort
import matplotlib.pyplot as plt

class DB_MAP:

    __db_traslation_x = 0
    __db_traslation_y = 0
    __db_imgs = 0

    db_path = ''
    scale_x = 0
    scale_y = 0

    db_loc_x = []
    db_loc_y = []

    __fig = 0
    __ax = 10
    
    # def test_print(self):
        # print(self.__ax)

    def get_ax(self):
        return self.__ax
    
    def get_db_imgs(self):
        return self.__db_imgs

    def db_map(self):
        #이미지 디렉토리까지가서 읽기.
        os.chdir(self.db_path)
        self.__db_imgs = os.listdir(self.db_path)
        self.__db_imgs = natsort.natsorted(self.__db_imgs)

        #DB좌표 만들기.
        cor_list = []
        i = 0
        for data in self.__db_imgs :
            #.jpg없에기
            data = data.replace('.JPG','')
            tmp = data.split('_')
            cor_list.append([])
            for t in tmp :
                t = int(t)
                cor_list[i].append(t)
            i = i + 1

        self.__db_traslation_x, self.__db_traslation_y = zip(*cor_list)
        
        # self.__db_traslation_x ,self.__db_traslation_y = self.normalize(self.__db_traslation_x , self.__db_traslation_y, scale_x, scale_y)
        self.normalize(self.__db_traslation_x , self.__db_traslation_y, self.scale_x, self.scale_y)

        self.db_loc_x.append(self.__db_traslation_x)
        self.db_loc_y.append(self.__db_traslation_y)
        # print("self.db_loc_x:\n", self.db_loc_x[0][1])
        # print("self.db_loc_y:\n", self.db_loc_y[0][1])
        

        self.__fig, self.__ax = plt.subplots(figsize=(18, 6))
        self.__ax.scatter(self.__db_traslation_x,self.__db_traslation_y, 100, c='blue')
        self.__ax.set_xlim([-20, 850])
        self.__ax.set_ylim([-20, 350])
        
        for k in range(0, len(self.db_loc_x[0])):
            self.Annotate(f"{k}", (self.db_loc_x[0][k], self.db_loc_y[0][k]), 10)

        # return self.__db_traslation_x, self.__db_traslation_y, self.__db_imgs


    # def db_matching(self, db_traslation_x, db_traslation_y, answer_cor, answer_range) :
    def db_matching(self, answer_cor, answer_range) :
        x_range_left = answer_cor[0] - answer_range
        x_range_right = answer_cor[0] + answer_range
        y_range_bottom = answer_cor[1] - answer_range
        y_range_top =  answer_cor[1] + answer_range

        db_answer_index = []
        for i in range(0,len(self.__db_traslation_x)) :
            if x_range_left < self.__db_traslation_x[i] and self.__db_traslation_x[i] < x_range_right and y_range_bottom < self.__db_traslation_y[i] and self.__db_traslation_y[i] < y_range_top :
                db_answer_index.append(i)
                
        print("matching DB index is",db_answer_index)
        # for i in db_answer_index :
        #     print(i , " = ", db_traslation_x[i], " , ", db_traslation_y[i])
        return db_answer_index


    def normalize(self,cor_x,cor_y, scale_x, scale_y):
        self.__db_traslation_x = list(cor_x)
        self.__db_traslation_y = list(cor_y)
        for i in range(len(self.__db_traslation_x))  :
            self.__db_traslation_x[i] = (self.__db_traslation_x[i] * scale_x)
        

        for i in range(len(self.__db_traslation_y))  :
            self.__db_traslation_y[i] = (self.__db_traslation_y[i] * scale_y)

        # return x,y

    def Scatter(self, answer1, answer2, val_s, color):
        self.__ax.scatter(answer1, answer2, s = val_s, c = color)

    def Annotate(self, Q_D, val_xy, val_fontsize):
        self.__ax.annotate(Q_D, xy=val_xy, fontsize = val_fontsize)
        
    def Legend(self, scat_all, scat_max):
        str1 = "DB map"
        str2 = "Tracking Result"
        str3 = "Matching -> Pose Estimation (max)"
        str4 = "Matching -> Pose Estimation (all)"
                
        if(scat_all == 1):
            if(scat_max == 1):
                self.__ax.legend((str1, str2, str4, str3), loc="lower right")
            else:
                self.__ax.legend((str1, str2, str4), loc="lower right")
        else:
            if(scat_max == 1):
                self.__ax.legend((str1, str2, str3), loc="lower right")
            else:
                self.__ax.legend((str1, str2), loc="lower right")