import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

class Radius_Match:

    draw_circle = 0
    radiusMatching = 1
    nfeatures = 0

    radius = 0
    K = 0
    K_d = 0
    K_list = []
    K_d_list = []
    
    E_update_radius = np.eye(3)
        
    flags = 0
    
    R = []
    t = []
    RT = []
    RT_update = np.eye(4)
    
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
        self.RT = np.array(RT).reshape(4, 4)

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
        
        self.R = np.array(R).reshape(3, 3)
        self.t = np.array(t).reshape(3, 1)
        
    
    def radius_match(self, img1, img2):
        F_empty = 0

        # start = time.time()
        sift = cv2.SIFT_create(self.nfeatures)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        # bf = cv2.BFMatcher()

        gray_q = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray_d = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 

        kp_q, des_q = sift.detectAndCompute(gray_q, None)
        kp_d, des_d = sift.detectAndCompute(gray_d, None)
        
        matches = bf.knnMatch(des_q, des_d, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good_matches.append(m)
        
        # good_matches = bf.match(des_q, des_d)
        
        img_matches = cv2.drawMatches(gray_q, kp_q, gray_d, kp_d, good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # matchesThickness = 3,

        if(self.radiusMatching == 0):
            # # motion vector
            # motion_vectors = []
            # for match in new_good_matches:
            #     q_kp = new_kp_q[match.queryIdx].pt
            #     d_kp = new_kp_d[match.trainIdx].pt
                
            #     motion_vector = (d_kp[0] - q_kp[0], d_kp[1] - q_kp[1])
            #     motion_vectors.append(motion_vector)
            
            src = np.float32([kp_q[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
            dst = np.float32([kp_d[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
            
            
            
            F, mask = cv2.findFundamentalMat(src, dst, cv2.RANSAC, 5.0)
            
            E = self.K_d.T @ F @ self.K
            
            # U,S,V = np.linalg.svd(E)
            # S = [[1,0,0],[0,1,0],[0,0,0]]
            # E = U @ S @ V
            
            _, self.R, self.t, mask = cv2.recoverPose(E, src, dst)
        
            self.makeRTMatrix(self.R, self.t.ravel())
            
            self.RT_update = self.RT_update @ self.RT
            
            self.recoverRTMatrix(self.RT_update)

            if(self.draw_circle == 1):
                img_matches_clone = img_matches.copy()

                # print(len(kp_q)) # juno

                # for i in range(0, len(kp_d)):
                for i in range(0, len(kp_q)):

                    r = np.random.randint(255)
                    g = np.random.randint(255)
                    b = np.random.randint(255)
                    color = (b, g, r)
                    
                    qx = round(kp_q[i].pt[0])
                    qy = round(kp_q[i].pt[1])
                    center_q = (qx, qy)
                    
                    dx = round(img1.shape[1] + qx)
                    dy = qy
                    center_d = (dx, dy)
        
                    # query
                    circle_img_matches = cv2.circle(img_matches_clone, center_q, self.radius, color, 1)
                    #db
                    circle_img_matches = cv2.circle(img_matches_clone, center_d, self.radius, color, 1)
                    
                # Display the matches
                plt.figure(100)
                plt.imshow(circle_img_matches)
                plt.title('sift matching w circle')

                plt.figure(200)
                plt.imshow(img_matches)
                plt.title('sift matcing w/o circle')

                # end_2 = time.time()
                # print(f"time - drawing circle O: {end_2 - start: .5f} sec")
                
                plt.show()

            
            return img_matches, good_matches, self.R, self.t
        else:
            new_kp_q = []
            new_kp_d = []

            for k in range(0, len(good_matches)):
                
                diff_x = kp_d[good_matches[k].trainIdx].pt[0] - kp_q[good_matches[k].queryIdx].pt[0]
                diff_y = kp_d[good_matches[k].trainIdx].pt[1] - kp_q[good_matches[k].queryIdx].pt[1]
                
                if((diff_x**2 + diff_y**2)**(1/2) <= self.radius):
                    temp_q = cv2.KeyPoint(kp_q[good_matches[k].queryIdx].pt[0], kp_q[good_matches[k].queryIdx].pt[1], kp_q[good_matches[k].queryIdx].size, kp_q[good_matches[k].queryIdx].angle, kp_q[good_matches[k].queryIdx].response, kp_q[good_matches[k].queryIdx].octave)
                    temp_d = cv2.KeyPoint(kp_d[good_matches[k].trainIdx].pt[0], kp_d[good_matches[k].trainIdx].pt[1], kp_d[good_matches[k].trainIdx].size, kp_d[good_matches[k].trainIdx].angle, kp_d[good_matches[k].trainIdx].response, kp_d[good_matches[k].trainIdx].octave)
                    new_kp_q.append(temp_q)
                    new_kp_d.append(temp_d)

            new_kp_q = tuple(new_kp_q)
            new_kp_d = tuple(new_kp_d)

            new_des_q = sift.compute(gray_q, new_kp_q)
            new_des_d = sift.compute(gray_d, new_kp_d)

            new_des_q = np.array(new_des_q[1])
            new_des_d = np.array(new_des_d[1])

            # new_matches = bf.knnMatch(new_des_q, new_des_d, k=2)

            # new_good_matches = []
            # for m, n in new_matches:
            #     if m.distance < 0.9 * n.distance:
            #         new_good_matches.append(m)
              
            new_good_matches = bf.match(new_des_q, new_des_d)
                    
            # # motion vector
            # motion_vectors = []
            # for match in new_good_matches:
            #     q_kp = new_kp_q[match.queryIdx].pt
            #     d_kp = new_kp_d[match.trainIdx].pt
                
            #     motion_vector = (d_kp[0] - q_kp[0], d_kp[1] - q_kp[1])
            #     motion_vectors.append(motion_vector)
                
            # motion_vectors_norm = np.linalg.norm(motion_vectors)
            
            # print(motion_vectors)
            # print(motion_vectors_norm)
                
            # Draw the matches
            new_img_matches = cv2.drawMatches(gray_q, new_kp_q, gray_d, new_kp_d, new_good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # matchesThickness = 3, 
            new_img_matches_clone = new_img_matches.copy()
            
            # end = time.time()
            # print(f"time - drawing circle X: {end - start: .5f} sec")

            # draw circle
            if(self.draw_circle == 1):
                for i in range(0, len(new_kp_d)):

                    r = np.random.randint(255)
                    g = np.random.randint(255)
                    b = np.random.randint(255)
                    color = (b, g, r)
                    
                    qx = round(new_kp_q[i].pt[0])
                    qy = round(new_kp_q[i].pt[1])
                    center_q = (qx, qy)
                    
                    dx = round(img1.shape[1] + qx)
                    dy = qy
                    center_d = (dx, dy)

                    # query
                    circle_new_img_matches = cv2.circle(new_img_matches_clone, center_q, self.radius, color, 3)
                    #db
                    circle_new_img_matches = cv2.circle(new_img_matches_clone, center_d, self.radius, color, 3)
                    
                    # query
                    circle_img_matches = cv2.circle(img_matches, center_q, self.radius, color, 3)
                    #db
                    circle_img_matches = cv2.circle(img_matches, center_d, self.radius, color, 3)
                    
                # Display the matches
                plt.figure(100)
                plt.imshow(circle_img_matches)
                # plt.imshow(new_img_matches)
                plt.title('before radius matching w circle')

                plt.figure(200)
                plt.imshow(circle_new_img_matches)
                plt.title('after radius matcing w circle')

                plt.figure(300)
                plt.imshow(new_img_matches)
                plt.title('radius matching w/o circle')
                
                # end_2 = time.time()
                # print(f"time - drawing circle O: {end_2 - start: .5f} sec")
                
                plt.show()
                plt.close()
                
            src = np.array([new_kp_q[match.queryIdx].pt for match in new_good_matches]).reshape(-1, 1, 2)
            dst = np.array([new_kp_d[match.trainIdx].pt for match in new_good_matches]).reshape(-1, 1, 2)
            
            # Compute the Fundamental matrix using RANSAC
            F, mask = cv2.findFundamentalMat(src, dst)  
            
            # if(F == None):
            #     print("*** F is None ***")
            #     F_empty = 1
            #     return new_img_matches, new_matches, 0, 0, E_update, F_empty
            
            E = self.K.T @ F @ self.K_d
        
            _, self.R, self.t, mask = cv2.recoverPose(E, src, dst)
            
            # self.makeRTMatrix(self.R, self.t.ravel())
            
            # self.RT_update = self.RT_update @ self.RT
            
            # self.recoverRTMatrix(self.RT_update)

            return new_img_matches, new_good_matches, self.R, self.t

    def moving_radius_match(self, img1, img2, radius, K, K_d, draw_circle=0, radiusMatching = 1, nfeatures = 1000):
        # start = time.time()
        sift = cv2.SIFT_create(nfeatures)
        bf = cv2.BFMatcher(cv2.NORM_L2)

        new_gray_q = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        new_gray_d = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 

        kp_q, new_des_q = sift.detectAndCompute(new_gray_q, None)
        kp_d, new_des_d = sift.detectAndCompute(new_gray_d, None)

        matches = bf.match(new_des_q, new_des_d)
        matches = sorted(matches, key=lambda x: x.distance)

        img_matches = cv2.drawMatches(new_gray_q, kp_q, new_gray_d, kp_d, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # matchesThickness = 3, 


        if(radiusMatching == 0):
            src = np.float32([kp_q[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
            dst = np.float32([kp_d[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
            
            F, mask = cv2.findFundamentalMat(src, dst, cv2.RANSAC, 5.0)
            
            E = K.T @ F @ K_d
            
            _, R, t, mask = cv2.recoverPose(E, src, dst) # , focal=focal_, pp=pp_
            
            if(draw_circle == 1):
                img_matches_clone = img_matches.copy()
                
                for i in range(0, len(kp_d)):

                    r = np.random.randint(255)
                    g = np.random.randint(255)
                    b = np.random.randint(255)
                    color = (b, g, r)
                    
                    qx = round(kp_q[i].pt[0])
                    qy = round(kp_q[i].pt[1])
                    center_q = (qx, qy)
                    
                    dx = round(img1.shape[1] + qx)
                    dy = qy
                    center_d = (dx, dy)
        
                    # query
                    circle_img_matches = cv2.circle(img_matches_clone, center_q, radius, color, 3)
                    #db
                    circle_img_matches = cv2.circle(img_matches_clone, center_d, radius, color, 3)
                    
                # Display the matches
                plt.figure(100)
                plt.imshow(circle_img_matches)
                plt.title('sift matching w circle')

                plt.figure(200)
                plt.imshow(img_matches)
                plt.title('sift matcing w/o circle')

                # end_2 = time.time()
                # print(f"time - drawing circle O: {end_2 - start: .5f} sec")
                
                plt.show()

            
            return img_matches, matches, R, t 

            
        else:
            new_kp_q = []
            new_kp_d = []

            for k in range(0, len(matches)):
                # diff_x = kp_d[matches[k].trainIdx].pt[0] - (kp_q[matches[k].queryIdx].pt[0] + T[0])
                # diff_y = kp_d[matches[k].trainIdx].pt[1] - (kp_q[matches[k].queryIdx].pt[1] + T[1]) 

                diff_x = kp_d[matches[k].trainIdx].pt[0] - (kp_q[matches[k].queryIdx].pt[0] + t[0])
                diff_y = kp_d[matches[k].trainIdx].pt[1] - (kp_q[matches[k].queryIdx].pt[1] + t[1])        
                
                if((diff_x**2 + diff_y**2)**(1/2) <= radius):
                    temp_q = cv2.KeyPoint(kp_q[matches[k].queryIdx].pt[0], kp_q[matches[k].queryIdx].pt[1], kp_q[matches[k].queryIdx].size, kp_q[matches[k].queryIdx].angle, kp_q[matches[k].queryIdx].response, kp_q[matches[k].queryIdx].octave)
                    temp_d = cv2.KeyPoint(kp_d[matches[k].trainIdx].pt[0], kp_d[matches[k].trainIdx].pt[1], kp_d[matches[k].trainIdx].size, kp_d[matches[k].trainIdx].angle, kp_d[matches[k].trainIdx].response, kp_d[matches[k].trainIdx].octave)
                    new_kp_q.append(temp_q)
                    new_kp_d.append(temp_d)

            new_kp_q = tuple(new_kp_q)
            new_kp_d = tuple(new_kp_d)

            new_des_q = sift.compute(new_gray_q, new_kp_q)
            new_des_d = sift.compute(new_gray_d, new_kp_d)

            new_des_q = np.array(new_des_q[1])
            new_des_d = np.array(new_des_d[1])

            new_matches = bf.match(new_des_q, new_des_d)

            new_matches = sorted(new_matches, key=lambda x: x.distance)
            
            # # find homography
            # src = np.array([new_kp_q[match.queryIdx].pt for match in new_matches])
            # dst = np.array([new_kp_d[match.trainIdx].pt for match in new_matches])
            
            # # R, t = get_Rt(src, dst, 301, (317., 252.))
            # R, t = get_Rt(src, dst, 301, (316., 251.))
            
            # # # normalize the translation vectors cuz last translation matrix vector is not 1 [[], [], [not 1]]
            # t[0][0] = t[0][0] / t[2][0]
            # t[1][0] = t[1][0] / t[2][0]
            # t[2][0] = t[2][0] / t[2][0]
            
            # Draw the matches
            new_img_matches = cv2.drawMatches(new_gray_q, new_kp_q, new_gray_d, new_kp_d, new_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # matchesThickness = 3, 
            new_img_matches_clone = new_img_matches.copy()
            
            # end = time.time()
            # print(f"time - drawing circle X: {end - start: .5f} sec")

            # draw circle
            if(draw_circle == 1):
                for i in range(0, len(new_kp_d)):

                    r = np.random.randint(255)
                    g = np.random.randint(255)
                    b = np.random.randint(255)
                    color = (b, g, r)
                    
                    qx = round(new_kp_q[i].pt[0])
                    qy = round(new_kp_q[i].pt[1])
                    center_q = (qx, qy)
                    
                    dx = round(img1.shape[1] + qx)
                    dy = qy
                    center_d = (dx, dy)

                    # query
                    circle_new_img_matches = cv2.circle(new_img_matches_clone, center_q, radius, color, 3)
                    #db
                    circle_new_img_matches = cv2.circle(new_img_matches_clone, center_d, radius, color, 3)
                    
                    # query
                    circle_img_matches = cv2.circle(img_matches, center_q, radius, color, 3)
                    #db
                    circle_img_matches = cv2.circle(img_matches, center_d, radius, color, 3)
                    
                # Display the matches
                plt.figure(1000)
                plt.imshow(img_matches)
                plt.title('before radius matching w circle')

                plt.figure(200)
                plt.imshow(circle_new_img_matches)
                plt.title('after radius matcing w circle')

                plt.figure(300)
                plt.imshow(new_img_matches)
                plt.title('radius matching w/o circle')
                
                # end_2 = time.time()
                # print(f"time - drawing circle O: {end_2 - start: .5f} sec")
                
                plt.show()
                
            ### find homography - method 1
            src = np.array([new_kp_q[match.queryIdx].pt for match in new_matches]).reshape(-1, 1, 2)
            dst = np.array([new_kp_d[match.trainIdx].pt for match in new_matches]).reshape(-1, 1, 2)
            
            # Compute the Fundamental matrix using RANSAC
            F, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 1.0)
            #print(F)
            # Compute the Essential matrix
            # E = np.dot(np.dot(K.T, F), K_d)
            E = K.T @ F @ K_d

            # Normalize the Essential matrix
            U, S, Vt = np.linalg.svd(E)
            S_new = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
            E_new = U @ S_new @ Vt
            
            E_update = np.dot(E_update,E_new)
            _, R, t, mask = cv2.recoverPose(E_update, src, dst)
            
            # print('R: \n', R)
            # print('t: \n', t)

            return new_img_matches, new_matches, R, t, E_update

    def get_Rt(self, src_pts, dst_pts, focal_, pp_):
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, focal=focal_, pp=pp_, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, focal=focal_, pp=pp_)
        
        return R, t

    def get_translation(self, t, translation_xy, flags, direction):
        x_translation = t[0] / t[2]
        y_translation = t[1] / t[2]
        
        if flags == 0:
            translation_xy[0] += x_translation * direction[1]
            translation_xy[1] += y_translation * direction[0]
        else:
            translation_xy[0] += y_translation * direction[0]
            translation_xy[1] += x_translation * direction[1]
        
        return translation_xy