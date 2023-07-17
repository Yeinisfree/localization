import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

class Tracking:
    __src_pts = 0
    __dst_pts = 0
    __mean_direction_vector = [0,0]
    __prev_mean_direction_vector = [0, 0]
    __angle = 0.0
    __R = 0
    __t = 0
    __E_update = np.eye(3)
    __RT_update = np.eye(4)
    __flags = 1
    __translation_xy = [0.0, 0.0]
    __suspectation = 0

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
    
    def get_mean_direction_vector(self):
        return self.__prev_mean_direction_vector
    
    def get_suspectation(self):
        return self.__suspectation
    
    def feature_detection(self, img, nfeatures):
        sift = cv2.SIFT_create(nfeatures)
        kp, des = sift.detectAndCompute(img, None)
        return kp, des

    def feature_matching(self, prev_kp, prev_des, curr_kp, curr_des, ratio=0.8):
        bf = cv2.BFMatcher()
        
        matches = bf.knnMatch(prev_des, curr_des, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)
                
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
    
    def remove_outliers_direction_mannual(self, threshold):
            
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

        return prev_pts_copy, next_pts_copy
    
    def detect_rotation(self, i, angle_th):
        if i == 1:
            self.__prev_mean_direction_vector = self.__mean_direction_vector
        cos_angle = np.dot(self.__prev_mean_direction_vector, self.__mean_direction_vector) / (np.linalg.norm(self.__prev_mean_direction_vector) * np.linalg.norm(self.__mean_direction_vector))
        if np.isnan(cos_angle):
            angle = 0
        else:
            angle = np.arccos(cos_angle) * 180 / np.pi
        
        if abs(angle) > angle_th:
            print("The rotation is suspected.")
            print(f"q: {i}")
            self.__suspectation = 1
            
        else:
            self.__suspectation = 0

        self.__prev_mean_direction_vector = self.__mean_direction_vector

    
    def get_translation_Essential(self, pose, src, dst, K, flags):
        
            
        E, mask = cv2.findEssentialMat(src, dst, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)


        _, R, t, mask = cv2.recoverPose(E, src, dst)
        
        
        RT = self.makeRTMatrix(R, t.ravel())
        
        self.__RT_update = self.__RT_update.dot(RT)
        
        R, t = self.recoverRTMatrix(self.__RT_update)
    
        print("flags: ", flags)
        
        
        
        if flags == 0:
            translation = t ### 확인 필요
            pose[0] += translation[0][0]
            pose[1] += translation[1][0]
        else:
            translation = -t ### 확인 필요
            pose[0] += translation[1][0]
            pose[1] += translation[0][0]
        
        return pose
    
    def initialize__RT_update(self):
        self.__RT_update = np.eye(4)
    
    def get_translation_homography(self, pose, src, dst, K, flags):
        
            
        H, _ = cv2.findHomography(src)


        _, R, t, _ = cv2.decomposeHomographyMat(H, np.eye(3))
        
        # 예상되는 카메라 위치 4곳에 대한 각각의 R,t가 나온다.
        
        R = np.array(R[0])
        t = np.array(t[0])
        
        
        RT = self.makeRTMatrix(R, t.ravel())
        
        self.__RT_update = self.__RT_update.dot(RT)
        
        R, t = self.recoverRTMatrix(self.__RT_update)
        
        translation = -t
        
        if flags == 0:    
            pose[0] += translation[0][0]
            pose[1] += translation[1][0]
        else:
            pose[0] += translation[1][0]
            pose[1] += translation[0][0]
        
        return pose
    