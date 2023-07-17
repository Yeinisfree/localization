import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import natsort


class Tracking:

    __src_pts = 0
    __dst_pts = 0
    __src_pts_filtered = 0
    __dst_pts_filtered = 0

    __mean_direction_vector = [0,0]

    __angle = 0.0
    
    __R = 0
    __t = 0
    __E_update = np.eye(3)
    __RT_update = np.eye(4)

    __curr_rot_flag = 0
    __flags = 1
    __direction = [1, 1]

    __prev_rot_flag = False
    __curr_rot_flag = False

    __translation_xy = [0.0, 0.0]
    
    __turn = 0 # If it is determined that it is rotated, 1

    def makeRTMatrix(self, R, t):
        RT = []
        RT.append(R[0][0])
        RT.append(R[0][1])
        RT.append(R[0][2])
        RT.append(t[0])
        RT.append(R[1][0])
        RT.append(R[1][1])
        RT.append(R[1][2])
        RT.append(t[1])
        RT.append(R[2][0])
        RT.append(R[2][1])
        RT.append(R[2][2])
        RT.append(t[2])
        RT.append(0.0)
        RT.append(0.0)
        RT.append(0.0)
        RT.append(1.0)
        RT = np.array(RT).reshape(4, 4)
        
        return RT

    def recoverRTMatrix(self, RT):
        R = []
        t = []
        
        R.append(RT[0][0])
        R.append(RT[0][1])
        R.append(RT[0][2])
        R.append(RT[1][0])
        R.append(RT[1][1])
        R.append(RT[1][2])
        R.append(RT[2][0])
        R.append(RT[2][1])
        R.append(RT[2][2])
        
        t.append(RT[0][3])
        t.append(RT[1][3])
        t.append(RT[2][3])
        
        R = np.array(R).reshape(3, 3)
        t = np.array(t).reshape(3, 1)
        
        return R, t

    def get_flags_val(self):
        return self.__flags
    
    def get_direction_val(self):
        return self.__direction

    def calc_optical_flow(self, img1, img2):
        lk_params = dict(winSize=(3, 3), # 21 21
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, mask=None, qualityLevel=0.01, minDistance=12)
        curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
        # next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, None)
        
        self.__src_pts = prev_pts[status == 1]
        self.__dst_pts = next_pts[status == 1]

        # return self.__src_pts, self.__dst_pts

    def feature_matching(self, img1, img2, nfeatures, th = 0.75):
        sift = cv2.xfeatures2d.SIFT_create(nfeatures)
        bf = cv2.BFMatcher()
        
        prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)
        curr_kp, curr_des = sift.detectAndCompute(curr_gray, None)
        
        matches = bf.knnMatch(prev_des, curr_des, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < th * n.distance:
                good_matches.append(m)
            
        # new_img_matches = cv2.drawMatches(img1, prev_kp, img2, curr_kp, good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        
        # plt.figure(123123)
        # plt.imshow(new_img_matches)
        # plt.show()
        
        src_pts = np.float32([prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        
        return src_pts, dst_pts
        
    def remove_outliers_length(self, src, dst):
        
        motion_vectors =  src - dst
        
        # Depending on vectgor's length
        lengths = np.sqrt(np.sum(motion_vectors**2, axis=1))
        threshold = np.mean(lengths) + np.std(lengths)
        mask = lengths < threshold
        
        if len(src[mask]) < 20 or len(dst[mask]) < 20:
            return src, dst
        
        return src[mask], dst[mask]
        
    def remove_outliers_direction(self, threshold):
        
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
            # mask = np.abs(dot_products) > threshold
            mask = dot_products > threshold
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

    def remove_outliers_direction_auto(self, src, dst,max_iterations=500):
        
        if len(src) < 10 or len(dst) < 10:
            print("Not enough points to estimate fundamental matrix")
            return src, dst

        prev_pts_copy = src.copy()
        next_pts_copy = dst.copy()
        motion_vectors = prev_pts_copy - next_pts_copy
        direction_vector = motion_vectors / np.linalg.norm(motion_vectors, axis=1, keepdims=True)

        # Estimate the optimal threshold using cosine similarity between all pairs of motion vectors
        dot_products = np.dot(direction_vector, direction_vector.T)
        hist, bins = np.histogram(dot_products, bins=100)
        threshold = bins[np.argmax(hist)]
        np.delete(bins, np.argmax(hist))

        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            mean_direction_vector = np.mean(direction_vector, axis=0)
            mean_direction_vector = mean_direction_vector / np.linalg.norm(mean_direction_vector)

            dot_products = np.dot(direction_vector, mean_direction_vector.T)
            mask = dot_products > threshold
            outliers = np.where(mask == False)

            if len(outliers[0]) == 0:
                break

            nan_mean_vector = np.mean(direction_vector[mask])/np.linalg.norm(np.mean(direction_vector[mask]))

            if np.isnan(nan_mean_vector).any():
                threshold = bins[np.argmax(hist)]
                np.delete(bins, np.argmax(hist))
                prev_pts_copy = src.copy()
                next_pts_copy = dst.copy()
                motion_vectors = prev_pts_copy - next_pts_copy
                direction_vector = motion_vectors / np.linalg.norm(motion_vectors, axis=1, keepdims=True)
                continue

            prev_pts_copy = prev_pts_copy[mask]
            next_pts_copy = next_pts_copy[mask]
            direction_vector = direction_vector[mask]

            if len(prev_pts_copy) < 30 and len(next_pts_copy) < 30:
                # reset the threshold to the initial value and restart the process
                threshold = bins[np.argmax(hist)]
                np.delete(bins, np.argmax(hist))
                prev_pts_copy = src.copy()
                next_pts_copy = dst.copy()
                motion_vectors = prev_pts_copy - next_pts_copy
                direction_vector = motion_vectors / np.linalg.norm(motion_vectors, axis=1, keepdims=True)
                continue


        if iteration >= max_iterations:
            # print("Exceeded maximum iterations without convergence.")
            pass

        self.__mean_direction_vector = mean_direction_vector
        
        # print(prev_pts_copy)
        # print(next_pts_copy)
        # print(mean_direction_vector)
        # exit(0)
        
        return prev_pts_copy, next_pts_copy


    def get_angle(self):
        reference = (1, 0)
        dot_product = self.__mean_direction_vector[0] * reference[0] + self.__mean_direction_vector[1] * reference[1]
        self.__angle = math.acos(dot_product)
        self.__angle =  self.__angle * 180 / math.pi
        
        # return self.__angle



    # def get_Rt(self, focal_, pp_, E_update):
    def get_Rt(self, K):
        E, mask = cv2.findEssentialMat(self.__src_pts, self.__dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        _, self.__R, self.__t, mask = cv2.recoverPose(E, self.__src_pts, self.__dst_pts)
        
        RT = self.makeRTMatrix(self.__R, self.__t.ravel())
    
        self.__RT_update = self.__RT_update.dot(RT)
        
        self.__R, self.__t = self.recoverRTMatrix(self.__RT_update)
        
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
        
    def get_translation_homography(self, xy, src, dst, angle_th, K, flags):
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        
        self.__angle = math.atan2(M[0,1], M[0,0]) * 180/math.pi
        
        if abs(self.__angle) > angle_th:
            self.__angle_discriminator = 1
            
        E, mask = cv2.findEssentialMat(src, dst, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        _, R, t, mask = cv2.recoverPose(E, src, dst)
        
        
        RT = self.makeRTMatrix(R, t.ravel())
        
        self.__RT_update = self.__RT_update.dot(RT)
        
        R, t = self.recoverRTMatrix(self.__RT_update)
        
        # translation = t
        translation = -t
        
        if flags == 0:    
            xy[0] += translation[0][0] * 5.5
            xy[1] += translation[1][0] * 2
        else:
            xy[0] += translation[1][0] * 5.5
            xy[1] += translation[0][0] * 3
        
        return xy
        

    # def get_translation(self, t, translation_xy, flags, direction):
    def get_translation(self):
        # x_translation = self.__t[0][0] / self.__t[2][0]
        # y_translation = self.__t[1][0] / self.__t[2][0]
        
        # if self.__flags == 0:
        #     self.__translation_xy[0] += x_translation * self.__direction[1]
        #     self.__translation_xy[1] += y_translation * self.__direction[0]
        # else:
        #     self.__translation_xy[0] += y_translation * self.__direction[0]
        #     self.__translation_xy[1] += x_translation * self.__direction[1]
        
        if(self.__t[1][0] < 0):
            self.__t = -self.__t
        
        x_translation = self.__t[0][0]
        y_translation = self.__t[1][0]
        
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




