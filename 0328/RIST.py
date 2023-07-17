import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

import radius_match_essential
import create_map
import tracking
import fl_estimator
import os

class RIST():

    __query_imgs = []
    __max_t = [0, 0, 0]
    __db_img_answer = []

    __translation_xy = [0, 0]

    __answer_translation = [0, 0]
    __answer_translation_temp = [0, 0]
    __max_answer_translation = [0, 0]

    __rm_max_match_length = 0
    
    __focal_optim = 0
    
    __ratio = 0.8
    
    __db_idx_answer = 0


    # 초기화
    def __init__(self) -> None:
        
        self.__part = int(input("Part : "))
        self.__answer = int(input("쿼리 영상의 인덱스를 입력하시오 : "))
        self.__answer_range = float(input("Range : "))

        self.track = tracking.Tracking()
        self.rm = radius_match_essential.Radius_Match()
        self.cm = create_map.DB_MAP()
        
        self.import_val() # yaml 파일로부터 변수를 불러오는 함수
        self.import_img() # path에서 이미지 불러오는 함수
    
    # 찾고자 하는 q 순서 반환
    def get_answer(self) -> int:
        return self.__answer
    
    # yaml 파일로부터 변수를 불러오는 함수
    def import_val(self) -> None: 
        with open('part' + str(self.__part) + '.yaml', 'r') as f:
            self.__data = yaml.safe_load(f)

        self.__rmvoutlier_th = self.__data['rmvoutlier_th']
        self.__angle_th = self.__data['angle_th']
        self.__num_images = self.__data['num_images']
        self.__focal_optim = self.__data['focal_optim']
        
        # Create MAP
        self.cm.scale_x = self.__data['scale_x']
        self.cm.scale_y = self.__data['scale_y']
        self.cm.db_path = self.__data['db_path']

        self.__db_path = self.cm.db_path
        self.__q_path = self.__data['q_path']
        # self.__num_images = self.get_num_images(self.__q_path)
        
        self.__undistortion = self.__data['undistortion'] 
        self.__scat_all_db = self.__data['scat_all_db']
        self.__scat_max_db = self.__data['scat_max_db']

        # Matching
        self.rm.radius = self.__data['radius']
        
        # 전면 렌즈
        self.rm.K_d = np.array([[784.357 , 0.0  , 1383.291],
                                [0.0  , 787.542 , 1353.700],
                                [0.0  , 0.0  , 1.0]])
        # 후면 렌즈
        # self.rm.K_d = np.array([[802.806 , 0.0  , 1368.866],
        #                         [0.0  , 799.976 , 1388.745],
        #                         [0.0  , 0.0  , 1.0]])
        
        self.rm.draw_circle = self.__data['draw_circle']
        self.rm.radiusMatching = self.__data['radiusMatching']
        self.rm.nfeatures = self.__data['nfeatures']

    # 파일 경로로부터 이미지 불러와서 저장
    def import_img(self) -> None:
        self.rm.K_d = np.array([[1.88, 0.0, 1376.31],
                             [0.0, 1.88, 1375.22],
                             [0.0, 0.0, 1.0]])
        
        if(self.__undistortion == 1):
            """undistort"""
            for i in range(0, self.__num_images):
                # img = cv2.imread(self.__q_path + 'undistort_' + str(i) + ".jpg")
                img = cv2.imread(self.__q_path + str(i) + ".jpg")
                self.__query_imgs.append(img)
            
            self.__query_size = (self.__query_imgs[0].shape[0], self.__query_imgs[0].shape[1])
            
            # undistortion images camera intrinsic parameter
            self.rm.K = np.array([[301.39596558, 0.0, 316.70672662],
                            [0.0, 300.95941162, 251.54445701],
                            [0.0, 0.0, 1.0]])
            
            ### Focal Length Estimation ###
            if(self.__focal_optim == 1):
                for i in range(0, self.__num_images-1):
                    new_K = fl_estimator.FL_estimator(self.rm.K, self.__query_imgs[i], self.__query_imgs[i+1])
                    self.rm.K_list.append(new_K)
                    
                    if(i == self.__num_images-2):
                        self.rm.K_list.append(new_K)
            ###############################
            
        else:
            """distort"""
            for i in range(0, self.__num_images):
                img = cv2.imread("/home/aaron/RIST/dataset/query_distort/" + str(i) + ".jpg")
                self.__query_imgs.append(img)
                
            self.__query_size = (self.__query_imgs[0].shape[0], self.__query_imgs[0].shape[1])

            # distortion images camera intrinsic parameter
            # self.rm.K = np.array([[301.867624408757 , 0.0                , 317.20235900477695],
            #                     [0.0              , 301.58768437338944 , 252.0695806789168],
            #                     [0.0              , 0.0                , 1.0]])
            self.rm.K = np.array([[301.39596558, 0.0, 316.70672662],
                         [0.0, 300.95941162, 251.54445701],
                         [0.0, 0.0, 1.0]])


            ### Focal Length Estimation ###
            if(self.__focal_optim == 1):
                for i in range(0, self.__num_images-1):
                    new_K = fl_estimator.FL_estimator(self.rm.K, self.__query_imgs[i], self.__query_imgs[i+1])
                    self.rm.K_list.append(new_K)
                    
                    if(i == self.__num_images-1):
                        self.rm.K_list.append(new_K)
            ###############################
            
        ### Focal Length Estimation ###
        if(self.__focal_optim == 1):
            file_list = os.listdir(self.__db_path)
            len_db = len(file_list)
            
            db_temp = []
            for i in range(0, len_db):
                img = cv2.imread(self.__db_path + file_list[i])
                img = cv2.resize(img, (self.__query_size[1], self.__query_size[0]))
                db_temp.append(img)
                
            for i in range(0, len_db-1):
                new_K = fl_estimator.FL_estimator(self.rm.K_d, db_temp[i], db_temp[i+1])
                self.rm.K_d_list.append(new_K)
                
                if(i == len_db-2):
                    self.rm.K_d_list.append(new_K)
        ###############################

    # def tracking(self, i) -> None:

    #     img1 = self.__query_imgs[i-1]
    #     img2 = self.__query_imgs[i]

    #     self.track.calc_optical_flow(img1, img2)
        
    #     self.track.remove_outliers_direction(self.__rmvoutlier_th)
        
    #     self.track.remove_outliers_length()
        
    #     self.track.get_mean_direction_vector()
        
    #     self.track.get_angle()
        
    #     self.track.get_Rt(self.rm.K)

    #     self.track.detect_rotation(self.__angle_th)
    
    #     self.__translation_xy = self.track.get_translation()

    def tracking(self, i) -> None:
        img1 = self.__query_imgs[i-1]
        img2 = self.__query_imgs[i]
        
        self.track.__src_pts, self.track.__dst_pts = self.track.feature_matching(img1, img2, self.rm.nfeatures, self.__ratio)
        
        self.track.__src_pts, self.track.__dst_pts = self.track.remove_outliers_direction_auto(self.track.__src_pts, self.track.__dst_pts)
        
        self.track.__src_pts, self.track.__dst_pts = self.track.remove_outliers_length(self.track.__src_pts, self.track.__dst_pts)
        
        if(self.track.get_flags_val() == 0):
            T_flags = 1
        elif(self.track.get_flags_val() == 1):
            T_flags = 0
        
        self.__translation_xy = self.track.get_translation_homography(self.__translation_xy, self.track.__src_pts, self.track.__dst_pts, self.__angle_th, self.rm.K, self.track.get_flags_val()) # self.track.get_flags_val()

    def matching(self , i) -> None:

        # MATCHING ANSWER_QUERY WITH DB

        # if(self.track.__turn == 1):
        #     self.__answer_range *= 1.5
        
        self.__db_search_index = self.cm.db_matching(self.__translation_xy , self.__answer_range)

        self.cm.Scatter(self.__translation_xy[0] , self.__translation_xy[1], 180, 'red')
        # self.cm.Annotate(f"{i}", (self.__translation_xy[0] , self.__translation_xy[1]), 10)

        if(self.track.get_flags_val() == 0):
            T_flags = 1
        elif(self.track.get_flags_val() == 1):
            T_flags = 0

        # self.rm.flags = T_flags
        
        self.__rm_max_match_length = 0 ##############################################

        ######### db search idx #########
        # if(i % 2 == 1):
        #     self.__cnt+=1
        #     self.__db_search_index = [self.__cnt-1, self.__cnt, self.__cnt+1, self.__cnt+2, self.__cnt+3, self.__cnt+4, self.__cnt+5, self.__cnt+6, self.__cnt+7]
        # else:
        #     self.__db_search_index = [self.__cnt-1, self.__cnt, self.__cnt+1, self.__cnt+2, self.__cnt+3, self.__cnt+4, self.__cnt+5, self.__cnt+6, self.__cnt+7]
        
        # self.__db_search_index = [13, 14, 15, 16, 17, 18] # db idx 조작
        
        # for k in range(0, self.__num_images):
        #     self.__db_search_index.append(k)
        
        # self.__cnt+=1
        # self.__db_search_index = [self.__cnt-1, self.__cnt, self.__cnt+1, self.__cnt+2, self.__cnt+3, self.__cnt+4, self.__cnt+5, self.__cnt+6, self.__cnt+7, self.__cnt+8, self.__cnt+9]
        
        # print("db idx: ", self.__db_search_index)
        # print("query idx: ", i)
        #################################

        matched_images = []

        for j in self.__db_search_index:
            # db_img = cv2.imread(db_path + str(db_imgs[j]))
            db_img = cv2.imread(self.__db_path + str(self.cm.get_db_imgs()[j]))
            db_img = cv2.resize(db_img, (self.__query_size[1], self.__query_size[0]))
    
            ##
            if(self.__focal_optim == 1):
                self.rm.K = self.rm.K_list[i]
                self.rm.K_d = self.rm.K_d_list[j]
            ##
            new_img_matches, new_matches, rm_R, rm_t = self.rm.radius_match(self.__query_imgs[i], db_img)
            # new_img_matches, new_matches = self.rm.SIFT_Matching(self.__query_imgs[i], db_img)
            
            # if (F_empty == 1):
            #     continue
            
            # print("tranlsation matrix: \n", rm_t)###
            
            matched_images.append(new_img_matches)
            
            if self.__rm_max_match_length < len(new_matches):
                self.__rm_max_match_length = len(new_matches)
                self.__max_t[0] = rm_t[0]
                self.__max_t[1] = rm_t[1]
                self.__max_t[2] = rm_t[2]
                self.__db_idx_answer = j
                self.__db_img_answer.append(db_img)

            self.rm.t = self.__max_t #######
                
            # self.__answer_translation = self.__max_answer_translation.copy() # 이전 q에 위치
            db_x_temp = self.cm.db_loc_x[0][self.__db_idx_answer]
            db_y_temp = self.cm.db_loc_y[0][self.__db_idx_answer]
            db_loc_temp = [db_x_temp, db_y_temp]
            
            # print(f"DB {j}의", self.cm.db_loc_x[0][j], print(self.cm.db_loc_x[0][j]))
             
            self.__answer_translation_temp = self.rm.get_translation(self.rm.t, db_loc_temp, self.track.get_flags_val(), self.track.get_direction_val()) # T_flags
            
            ####################################################################################################
            if(self.__scat_all_db == 1):
                self.cm.Scatter(self.__answer_translation_temp[0], self.__answer_translation_temp[1], 90, 'green')
                x_temp = self.__answer_translation_temp[0].copy()
                y_temp = self.__answer_translation_temp[1].copy()
                self.cm.Annotate(f"{i}Q - {j}D", (x_temp, y_temp), 10)
            ###################################################################################################
        
        # images_to_display = cv2.vconcat(matched_images)
        
        # plt.figure(159)
        # plt.imshow(images_to_display)
        # plt.show()
        
        # plt.figure(112)
        # plt.imshow(new_img_matches)
        
        # result = cv2.hconcat([self.__query_imgs[i], self.__db_img_answer[-1]])
        # plt.figure(119)
        # plt.imshow(result)
        
        
        if i == self.__answer: 
            result = cv2.hconcat([self.__query_imgs[i], self.__db_img_answer[-1]])

            plt.subplots()
            plt.imshow(result)
        
        db_loc = [self.cm.db_loc_x[0][self.__db_idx_answer], self.cm.db_loc_y[0][self.__db_idx_answer]]
        
        # self.__max_answer_translation = self.rm.get_translation(self.__max_t , self.__max_answer_translation, self.track.get_flags_val(), self.track.get_direction_val()) # T_flags
        self.__max_answer_translation = self.rm.get_translation(self.__max_t , db_loc, self.track.get_flags_val(), self.track.get_direction_val())
        ####################################################################################################
        if(self.__scat_max_db == 1):
            # print("Pose before focal optim: \n", self.__max_answer_translation)s
            self.cm.Scatter(self.__max_answer_translation[0], self.__max_answer_translation[1], 50, 'purple') #yellow
            
            if(self.__scat_all_db == 0):
                x_temp = self.__max_answer_translation[0].copy()
                y_temp = self.__max_answer_translation[1].copy()
                # self.cm.Annotate(f"{i}Q - {j}D", (x_temp, y_temp), 6)
                # self.cm.Annotate(f"{i}Q", (x_temp, y_temp), 6)
        ####################################################################################################

        self.cm.Legend(self.__scat_all_db, self.__scat_max_db)
                
    
    # 경로 안에 이미지가 몇 개 있는지 출력하는 함수
    def get_num_images(self, q_path) -> int:

        valid_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]  # 이미지 파일 확장자

        # 폴더 내의 파일 목록 가져오기
        file_list = os.listdir(q_path)

        # 이미지 파일 개수 초기화
        num_images_ = 0

        # 폴더 내의 파일 목록 순회하며 이미지 파일 개수 카운트
        for file_name_ in file_list:
            ext = os.path.splitext(file_name_)[-1].lower()  # 파일 확장자 추출
            if ext in valid_extensions:
                num_images_ += 1
        
        num_images_ -= 1
        return num_images_