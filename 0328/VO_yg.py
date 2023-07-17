import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random as rand
import os, sys, copy, time
from pathlib import Path 
import re
import glob

inv = np.linalg.inv
det = np.linalg.det
svd = np.linalg.svd


# SVD: singular value decomposition
# 어떤 행렬 A가 있을 때 , 이 행렬을 세 개의 행렬 U,S,V로 분해하는 것
# 여기서 U와 V는 각각 직교행렬(orthogonal matrix)이며, S는 대각행렬(diagonal matrix)임
# S는 대각 성분이 0이 아닌 값들의 내림차순으로 정렬되어 있는데, 이를 특이값(singular value)라고 부름.
# SVD의 중요한 성질 중 하나는 이 특이값이 원래 행렬 A의 특성을 나타낸다는 것



# RIST Raspberry-PI Camera
K = np.array([[301.39596558, 0.0, 316.70672662],
                         [0.0, 300.95941162, 251.54445701],
                         [0.0, 0.0, 1.0]])
# # Fisheye Camera
# K = np.array([[370.28762825257195, 0.0, 507.91353092715286],
#                           [0.0, 370.4823749229323, 424.7687053137804],
#                           [0.0, 0.0, 1.0]])



K_inv = np.linalg.inv(K) 

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

class Cells:
    def __init__(self):
        self.pts = list()
        self.pairs = dict()

    def rand_pt(self):
        return rand.choice(self.pts)
class VisualOdometry:
    # =========================================================================================================================================================================================================================== #
    # Get Random 8 points from different regions in a Image using Zhang's 8x8 Grid
    # =========================================================================================================================================================================================================================== #
    def get_rand8(self, grid: np.array)-> list:            
        cells = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7)]
        rand_grid_index = rand.choices(cells, k = 8)   
        rand8 = list() 
        rand8_ = list()        
        for index in rand_grid_index:
            if grid[index].pts: 
                pt = grid[index].rand_pt()
                rand8.append(pt)
            else:
                index = rand.choice(cells)
                while not grid[index].pts or index in rand_grid_index:
                    index = rand.choice(cells) 
                pt = grid[index].rand_pt()
                rand8.append(pt)

            # -----> find the correspondence given point <----- #
            rand8_.append(grid[index].pairs[pt])
        return rand8, rand8_
    # =========================================================================================================================================================================================================================== #
    # Calculate Fundamental Matrix for the given * points from RANSAC
    # =========================================================================================================================================================================================================================== #  
    def calcualte_fundamental_matrix(self, pts_cf: np.array, pts_nf: np.array)-> list:
        F_CV,_ = cv2.findFundamentalMat(pts_cf,pts_nf,cv2.FM_8POINT)
        mat = []
        # ======================================================================================
        # 입력으로 들어온 대응되는 점들(pts_cf, pts_nf)을 이용하여 스케일 맞춤 및 중앙 정렬 수행
        # 각 점을 중앙으로 이동시키고, 이동 거리를 스케일로 나눠서 스케일을 맞추어 줌
        origin = [0.,0.]
        origin_ = [0.,0.]	
        origin = np.mean(pts_cf, axis = 0)
        origin_ = np.mean(pts_nf, axis = 0)	
        k = np.mean(np.sum((pts_cf - origin)**2 , axis=1, keepdims=True)**.5)
        k_ = np.mean(np.sum((pts_nf - origin_)**2 , axis=1, keepdims=True)**.5)
        k = np.sqrt(2.)/k
        k_ = np.sqrt(2.)/k_
        # ======================================================================================
        x = ( pts_cf[:, 0].reshape((-1,1)) - origin[0])*k
        y = ( pts_cf[:, 1].reshape((-1,1)) - origin[1])*k
        x_ = ( pts_nf[:, 0].reshape((-1,1)) - origin_[0])*k_
        y_ = ( pts_nf[:, 1].reshape((-1,1)) - origin_[1])*k_
        # ======================================================================================
        # 스케일이 맞춰진 대응되는 점들을 이용하여 A행렬을 만듦
        # A행렬의 마지막 특이벡터 V는 F를 구할 때 사용
        A = np.hstack((x_*x, x_*y, x_, y_ * x, y_ * y, y_, x,  y, np.ones((len(x),1))))	
        U,S,V = np.linalg.svd(A)
        F = V[-1]
        F = np.reshape(F,(3,3))
        # ======================================================================================
        # SVD를 사용하여 F행렬을 U,S,V 세 행렬로 분해
        # S행렬은 F의 sigular value들을 대각선에 오름차순으로 나열한 것
        # S행렬의 마지막 성분을 0으로 만들어주어 rank-2행렬로 변환
        U,S,V = np.linalg.svd(F)
        S[2] = 0
        F = U@np.diag(S)@V
        # ======================================================================================
        # 정규화를 위해 첫 번째 이미지의 특징점을 평균이 0, 표준편차가 sqrt(2)인 상태로 변환하는 T1
        # 두 번째 이미지의 특징점을 같은 방식으로 변환하는 T2 계산
        # T1, T2를 이용하여 F행렬 정규화
        # 마지막으로 F행렬의 마지막 원소로 나누어 정규화 -> homogeneous coordinates를 다루기 때문에 F행렬의 크기를 일정하게 유지하기 위해 필요 -> homogeneous coordinates에서의 마지막 값 w를 1로 
        T1 = np.array([[k, 0,-k*origin[0]], [0, k, -k*origin[1]], [0, 0, 1]])
        T2 = np.array([[k_, 0,-k_*origin_[0]], [0, k_, -k_*origin_[1]], [0, 0, 1]])
        F = T2.T @ F @ T1
        F = F / F[-1,-1]
        return F,F_CV

    # =========================================================================================================================================================================================================================== #
    # Estimate Fundamental Matrix from the given correspondences using RANSAC
    # =========================================================================================================================================================================================================================== #  
    def estimate_fundamental_matrix_RANSAC(self, pts1, pts2, matches, grid, epsilon = 0.05)-> list:
    
    # 첫 번째 이미지의 대응점인 pts1, 두 번째 이미지의 대응점인 pts2, 대응점의 매칭 결과 matches, 그리드 크기인 grid, 오차 범위인 epsilon
    # epsilon은 대응점이 Fundamental Matrix에 맞는 지 판단하는 기준이 되는 임계값    
    # max_inliners, F_best, S_in, confidence, N, count -> RANSAC 알고리즘을 구현하기 위해 사용
    # RANSAC을 N번 반복하며, 매 반복마다 pts1과 pts2로부터 무작위의 8개이 대응점을 선택하여 F 계산
    # F를 이용해 모든 대응점을 이용한 에러를 계산하고, 에러가 epsilon 이하인 대응점을 inliners로 판단.
    # inliners의 개수가 이전 반복에서 계산된 max_liners보다 크다면, max_liners를 현재 inliners의 개수로 갱신
    # F_best를 현재 F로 갱신
    # 그리고 RANSAC 반복 횟수인 count를 증가시킴
    # I_O_ration는 inliners 개수를 전체 대응점 개수로 나눈 비율
    # 이 비율을 8제곱한 값의 로그값을 구하고, confidence 값을 로그 값으로 나누어서 N을 계산
    # confidence = 알고리즘이 최적의 해를 찾아낼 확률
        max_inliers= 0
        F_best = []
        S_in = []
        confidence = 0.99
        N = sys.maxsize
        count = 0
        while N > count:
            S = []
            counter = 0
            x_1,x_2 = self.get_rand8(grid)
            F,F_b = self.calcualte_fundamental_matrix(np.array(x_1), np.array(x_2))
            ones = np.ones((len(pts1),1))
            x = np.hstack((pts1,ones))
            x_ = np.hstack((pts2,ones))
            e, e_ = x @ F.T, x_ @ F
            error = np.sum(e_* x, axis = 1, keepdims=True)**2 / np.sum(np.hstack((e[:, :-1],e_[:,:-1]))**2, axis = 1, keepdims=True)
            inliers = error<=epsilon
            counter = np.sum(inliers)
            if max_inliers <  counter:
                max_inliers = counter
                F_best = F 
            I_O_ratio = counter/len(pts1)
            if np.log(1-(I_O_ratio**8)) == 0: continue
            N = np.log(1-confidence)/np.log(1-(I_O_ratio**8))
            count += 1
        return F_best
    # =========================================================================================================================================================================================================================== #
    # Estimate Essential Matrix 
    # =========================================================================================================================================================================================================================== #
    def estimate_Essential_Matrix(self, K: np.array, F: np.array)-> np.array:	
        E = K.T @ F @ K
        U,S,V = np.linalg.svd(E)
        S = [[1,0,0],[0,1,0],[0,0,0]]
        E = U @ S @ V
        return E

    # =========================================================================================================================================================================================================================== #
    # Perform Linear Triangulation
    # =========================================================================================================================================================================================================================== #
    def linear_triangulation(self, K: np.array, C1: np.array, R1: np.array, C2: np.array, R2: np.array, pt: np.array, pt_: np.array)-> list:
        # 입력으로 카메라 내부 파라미터 K, 첫 번째 카메라의 중심 C1,과 회전 행렬R1, 두 번째 카메라의 중심 C2와 회전 행렬 R2, 그리고 두 이미지에서 매칭되는 특징점들 pt와 pt_
        
        # 카메라의 projection matrix P를 구하는 계산 과정
        # 카메라 내부 파라미터와 카메라의 포즈에 대한 homogeneous transformation matrix를 곱함  
        P1 = K @ np.hstack((R1, -R1 @ C1))
        P2 = K @ np.hstack((R2, -R2 @ C2))	
        
        # linear triangulation 수행
        # P1, P2와 점의 좌표를 이용하여 각 카메라에서의 점의 3D 위치를나타내는 4x3 크기의 행렬 A를 만듬
        # A에 대한 SVD를 수행하여 V행렬의 마지막 열을 이용하여 각 점의 3D 위치를 구함
        # 마지막 열의 원소를 나누어줌으로써 마지막 원소가 1이 되도록 보정
        # 보정된 4x1크기의 벡터를 X 리스트에 추가하고, 모든 매칭점에 대해 이 과정 반복
        X = []
        for i in range(len(pt)):
            x1 = pt[i]
            x2 = pt_[i]
            A1 = x1[0]*P1[2,:]-P1[0,:]
            A2 = x1[1]*P1[2,:]-P1[1,:]
            A3 = x2[0]*P2[2,:]-P2[0,:]
            A4 = x2[1]*P2[2,:]-P2[1,:]		
            A = [A1, A2, A3, A4]
            U,S,V = np.linalg.svd(A)
            V = V[3]
            V = V/V[-1]
            X.append(V)
        return X

    # =========================================================================================================================================================================================================================== #
    # Estimate the camera Pose
    # =========================================================================================================================================================================================================================== #    
    def camera_pose(self, K: np.array, E: np.array):
        # C1, C2, C3, C4: 카메라의 중심점(translation vector)을 나타내는 3X1 벡터
        # R1, R2, R3, R4: 카메라의 방향(rotation matrix)을 나타내는 3x3 벡터
        # P1, P2, P3, P4: 카메라 행렬(projection matrix)를 나타내는 3x4 벡터
        # 4개의 카메라 자세 추정
        # 회전 행렬의 determinant를 계산하여 양수인지 음수인지 확인
        # 회전 행렬은 물체의상대적인 방향을 나타내므로, 카메라의 위치가 상하대칭되는 경우는 음수가 됨
        # 이 경우 카메라 위치와 방향을 뒤집어줌으로써 올바른 카메라 자세를 얻을 수 있음
        
        W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        U,S,V = np.linalg.svd(E) # E = U @ S @ V (A = U @ Sigma @ V.T)
        
        poses = {}
        poses['C1'] = U[:,2].reshape(3,1)
        poses['C2'] = -U[:,2].reshape(3,1)
        poses['C3'] = U[:,2].reshape(3,1)
        poses['C4'] = -U[:,2].reshape(3,1)  
        poses['R1'] = U @ W @ V
        poses['R2'] = U @ W @ V 
        poses['R3'] = U @ W.T @ V
        poses['R4'] = U @ W.T @ V
        
        for i in range(4):
            C = poses['C'+str(i+1)]
            R = poses['R'+str(i+1)]
            if np.linalg.det(R) < 0:
                C = -C 
                R = -R 
                poses['C'+str(i+1)] = C 
                poses['R'+str(i+1)] = R
            I = np.eye(3,3)
            M = np.hstack((I,C.reshape(3,1)))
            poses['P'+str(i+1)] = K @ R @ M 
        
        # print(poses)
        # exit(0)
        
        return poses

    # =========================================================================================================================================================================================================================== #
    # Find the Rotation and Translation parametters
    # =========================================================================================================================================================================================================================== #
    def extract_Rot_and_Trans(self, R1: np.array, t: np.array, pt: np.array, pt_: np.array, K: np.array):
        # R1: 첫 번째 이미지의 회전행렬
        # t: 첫 번째 이미지의 평행이동행렬
        # pt: 첫 번째 이미지의 특징점 좌표
        # pt_: 두 번째 이미지의 대응하는 특징점 좌표
        # K: 내부 파라미터 행렬
        # 회전과 평행 이동 매개 변수를 추출하기 위해 3D triangulation을 사용
        # linear_triangulation()함수를 호출하여 3D점 추정
        # 추정된 점들을 반복하여 회전 행렬과 평행 이동 행렬을 사용하여 카메라의 움직임을 구함
        # 구한 움직임이 올바른지 확인하기 위해 모든 점을 루프 돌며, 첫 번째이미지와 두 번째 이미지 간의 점의 방향과 깊이가 올바른지 확인
        C = [[0],[0],[0]]
        R = np.eye(3,3)
        P = np.eye(3,4)
        P_ = np.hstack((R1,t))
        X1 = self.linear_triangulation(K, C, R,t,R1, pt, pt_)
        X1 = np.array(X1)	
        count = 0
        for i in range(X1.shape[0]):
            x = X1[i,:].reshape(-1,1)
            # 3D 점의 위치를 카메라 좌표계로 변환 후 카메라의 z축의 방향이 0보다 큰 경우에 count를 1씩 증가
            # np.subtract(x[0:3], t)는 3D 공간상의 점 x와 카메라 위치 t를 뺀 벡터
            # 따라서 R1[2]@np.subtract(x[0:3, t])는 카메라의 z축 방향과 점 x와 카메라 위치 t사이의 각도를 나타냄
            # 이 값이 0보다 큰 경우 카메라 앞쪽에 있다는 것을 의미
            if R1[2]@np.subtract(x[0:3],t) > 0 and x[2] > 0: count += 1 
        return count

    # =========================================================================================================================================================================================================================== #

if __name__=="__main__":
    # ----> Initialising Variables <----- # 
    Translation = np.zeros((3, 1))
    Rotation = np.eye(3)
    count = 0
    fig = plt.figure('Figure 1',figsize=(7,5))
    fig.suptitle('Visual Odometry')
    ax1 = fig.add_subplot(111)
    ax1.set_title('Visual Odometry Map')

    # ----> Loading images <----- #
    
    # img_files = sorted(glob.glob("/home/cgv/0306_0325/dataset/Part2_undistort/*.jpg"), key=get_order)
    img_files = sorted(glob.glob("/home/cgv/0326/newDataset/Camera_TEST/part3/*.jpg"), key=get_order)
    img_dim = cv2.imread(img_files[0]).shape
    y_bar, x_bar = np.array(img_dim[:-1])/8
    func = VisualOdometry()
    for i in range(1, len(img_files)):
        # ----> Load current and next frames <----- #
        key_frame_current = cv2.imread(img_files[i-1])
        key_frame_next = cv2.imread(img_files[i])
        current_frame = cv2.cvtColor(key_frame_current, cv2.COLOR_BGR2GRAY) 
        next_frame = cv2.cvtColor(key_frame_next, cv2.COLOR_BGR2GRAY)  

        # -----> Feature extraction using SIFT Algorithm <----- #
        sift = cv2.xfeatures2d.SIFT_create()    
        kp_cf,des_current = sift.detectAndCompute(current_frame,None)
        kp_nf,des_next = sift.detectAndCompute(next_frame,None)

        # -----> Extract the best matches <----- #
        best_matches = []
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_current,des_next,k=2)
        for m,n in matches:
            if m.distance < 0.5*n.distance: best_matches.append(m)

        # -----> Initialise the grids and points array variables <----- #
        point_correspondence_cf = np.zeros((len(best_matches),2)) # point correspoindence of current frame
        point_correspondence_nf = np.zeros((len(best_matches),2)) # point correspoindence of near frame
        grid = np.empty((8,8), dtype=object)
        grid[:,:] = Cells()

        # ----> Generating Zhang's Grid & extracting points from matches<----- #
        for i, match in enumerate(best_matches):
            j = int(kp_cf[match.queryIdx].pt[0]/x_bar)
            k = int(kp_cf[match.queryIdx].pt[1]/y_bar)
            grid[j,k].pts.append(kp_cf[match.queryIdx].pt)
            grid[j,k].pairs[kp_cf[match.queryIdx].pt] = kp_nf[match.trainIdx].pt

            point_correspondence_cf[i] = kp_cf[match.queryIdx].pt[0], kp_cf[match.queryIdx].pt[1]
            point_correspondence_nf[i] = kp_nf[match.trainIdx].pt[0], kp_nf[match.trainIdx].pt[1]

        F = func.estimate_fundamental_matrix_RANSAC(point_correspondence_cf, point_correspondence_nf, matches, grid, 0.05)                   # Estimate the Fundamental matrix #   
        E = func.estimate_Essential_Matrix(K, F)                                                                                            # Estimate the Essential Matrix #
        pose = func.camera_pose(K,E)                                                                                                        # Estimate the Posses Matrix #

        # -----> Estimate Rotation and Translation points <----- #
        # 입력된 이미지에 대한 카메라 포즈를 추정
        # 4개의 가능한 pose 후보 중에서 어떤 pose가 입력 이미지와 가장 일치하는 지를 결정
        # 입력된 rotation과 translation 후보, 그리고 2D 이미지 상의 대응되는 점들과 카메라 행렬을이용하여, 해당 후보로부터 대응되는 
        # 3D 점을 생성하고 이를 통해 3D점이 카메라 앞에 있는지, 뒤에 있는 지 판단하고, 이에 따라 해당 후보의 일치도를 결정
        # 일치도가 가장 높은 pose 후보가 결정되면, 그 포즈를 이용하여 translation과 rotation을 업데이트합니다.
        # 이를 통해 입력 이미지에 대한 추정된 pose를 얻을 수 있습니다.
        flag = 0
        for p in range(4):
            R = pose['R'+str(p+1)]
            T = pose['C'+str(p+1)]
            Z = func.extract_Rot_and_Trans(R, T, point_correspondence_cf, point_correspondence_nf, K)
            if flag < Z: flag, reg = Z, str(p+1)
        
        R = pose['R'+reg]
        t = pose['C'+reg] #####
        if t[2] < 0: t = -t
        x_cf = Translation[0]
        z_cf = Translation[2]
        Translation += Rotation.dot(t)
        Rotation = R.dot(Rotation)
        x_nf = Translation[0]
        z_nf = Translation[2]

        ax1.plot([-x_cf, -x_nf],[z_cf, z_nf],'o')
        if count%50 == 0: 
            plt.pause(1)
            # plt.savefig("./Output/"+str(count)+".png")
        else: plt.pause(0.001)
        count += 1
        print('# -----> Frame No:'+str(count),'<----- #')
    current_frame = next_frame

cv2.waitKey(0)
plt.show()
cv2.destroyAllWindows()