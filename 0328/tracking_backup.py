import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import natsort


class Tracking:

    __src_pts = 0
    __dst_pts = 0

    __mean_direction_vector = [0,0]

    __angle = 0.0
    
    __R = 0
    __t = 0
    __E_update = np.eye(3)

    __curr_rot_flag = 0
    __flags = 1
    __direction = [1, 1]

    __prev_rot_flag = False
    __curr_rot_flag = False

    __translation_xy = [0.0, 0.0]

    def get_flags_val(self):
        return self.__flags
    
    def get_direction_val(self):
        return self.__direction

    def calc_optical_flow(self, img1, img2):
        prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, mask=None, qualityLevel=0.01, minDistance=12)
        curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        
        self.__src_pts = prev_pts[status == 1]
        self.__dst_pts = next_pts[status == 1]

        # return self.__src_pts, self.__dst_pts


        
    def remove_outliers(self, threshold):
        
        motion_vectors =  self.__src_pts - self.__dst_pts
        direction_vector = motion_vectors / np.linalg.norm(motion_vectors, axis=1, keepdims=True)
        
        
        while True:
            self.__mean_direction_vector = np.mean(direction_vector, axis=0)
            self.__mean_direction_vector = self.__mean_direction_vector / np.linalg.norm(self.__mean_direction_vector)

            # A dot product of two vectors is a scalar value that represents the cosine similarity between the two vectors. 
            # If the dot product is positive, it means that the two vectors point in a similar direction. 
            # If the dot product is negative, it means that the two vectors point in opposite directions.
            # cosine similarity = dot(A, B) / (norm(A) * norm(B))
            dot_products = np.dot(direction_vector , self.__mean_direction_vector.T)
            #print(dot_products)
            mask = np.abs(dot_products) > threshold
            outliers = np.where(mask == False) 
            if len(outliers[0]) == 0:
                break
            # prev_pts = prev_pts[mask]
            # next_pts = next_pts[mask] 
            self.__src_pts = self.__src_pts[mask]
            self.__dst_pts = self.__dst_pts[mask] 
            
            direction_vector = direction_vector[mask]
            motion_vectors = motion_vectors[mask]
        
        
        # return self.__src_pts, self.__dst_pts



    def get_mean_direction_vector(self):
        motion_vectors = self.__src_pts - self.__dst_pts
        direction_vector = motion_vectors / np.linalg.norm(motion_vectors, axis=1, keepdims=True)
        self.__mean_direction_vector = np.mean(direction_vector, axis=0)
        self.__mean_direction_vector = self.__mean_direction_vector / np.linalg.norm(self.__mean_direction_vector)
        
        # return self.__mean_direction_vector



    def get_angle(self):
        reference = (1, 0)
        dot_product = self.__mean_direction_vector[0] * reference[0] + self.__mean_direction_vector[1] * reference[1]
        self.__angle = math.acos(dot_product)
        self.__angle =  self.__angle * 180 / math.pi
        
        # return self.__angle



    # def get_Rt(self, focal_, pp_, E_update):
    def get_Rt(self, focal_, pp_):
        E, mask = cv2.findEssentialMat(self.__src_pts, self.__dst_pts, focal=focal_, pp=pp_, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        self.__E_update = np.dot(self.__E_update,E)
        
        _, self.__R, self.__t, mask = cv2.recoverPose(self.__E_update, self.__src_pts, self.__dst_pts, focal=focal_, pp=pp_)
        
        # return self.__R, self.__t, self.__E_update



    # def detect_rotation(self, prev_rot_flag, flags, angle, threshold, direction):
    def detect_rotation(self, threshold):
        if self.__angle > threshold:
            self.__curr_rot_flag = True
        else:
            self.__curr_rot_flag = False
        
        if self.__prev_rot_flag == False and self.__curr_rot_flag == True:
            if self.__flags == 0: 
                self.__flags += 1
                self.__direction = self.get_direction(self.__mean_direction_vector, self.__direction)
            else: 
                self.__flags -= 1
                self.__direction = self.get_direction(self.__mean_direction_vector, self.__direction)
        
        self.__prev_rot_flag = self.__curr_rot_flag
        
        # return self.__curr_rot_flag, self.__flags, self.__direction
        

    def get_direction(self, mean_direction_vector, direction):
        if mean_direction_vector[0] > 0:
            direction[0] = 1
        else:
            direction[0] = -1
        if mean_direction_vector[1] > 0:
            direction[1] = 1
        else:
            direction[1] = -1
            
        return direction
        

    # def get_translation(self, t, translation_xy, flags, direction):
    def get_translation(self):
        x_translation = self.__t[0][0] / self.__t[2][0]
        y_translation = self.__t[1][0] / self.__t[2][0]
        
        if self.__flags == 0:
            self.__translation_xy[0] += x_translation * self.__direction[1]
            self.__translation_xy[1] += y_translation * self.__direction[0]
        else:
            self.__translation_xy[0] += y_translation * self.__direction[0]
            self.__translation_xy[1] += x_translation * self.__direction[1]

        return self.__translation_xy


    # def run(self, img1, img2, threshold1, focal_, pp_, threshold2):
    #     self.calc_optical_flow(img1, img2)
        
    #     self.remove_outliers(threshold1)
        
    #     self.get_mean_direction_vector()
        
    #     self.get_angle()
        
    #     self.get_Rt(focal_, pp_)

    #     self.detect_rotation(threshold2)
    
    #     self.translation_xy = self.get_translation()

    #     return self.translation_xy

    






    

    
    # def feature_detection(self, img, nfeatures):
    #     sift = cv2.xfeatures2d.SIFT_create(nfeatures)
    #     gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     kp, des = sift.detectAndCompute(gray1, None)
    #     return kp, des
        

    # def feature_matching(self, kp1, kp2, desc1, desc2, matcher):
        
    #     if matcher == "BF":
    #         bf = cv2.BFMatcher()
    #         matches = bf.knnMatch(desc1, desc2, k=2)
            
    #         good_matches = []
    #         for m, n in matches:
    #             if m.distance < 0.7 * n.distance:
    #                 good_matches.append(m)
            
    #         src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    #         dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
            
    #         return src_pts, dst_pts, good_matches
            
            
    #     elif matcher == "FLANN":
    #         pass




