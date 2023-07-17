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
        

    t = 0
    flags = 0
    
    def radius_match(self, img1, img2, E_update):
        F_empty = 0

        # start = time.time()
        sift = cv2.SIFT_create(self.nfeatures)
        bf = cv2.BFMatcher(cv2.NORM_L2)

        new_gray_q = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        new_gray_d = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 

        kp_q, new_des_q = sift.detectAndCompute(new_gray_q, None)
        kp_d, new_des_d = sift.detectAndCompute(new_gray_d, None)

        matches = bf.match(new_des_q, new_des_d)
        matches = sorted(matches, key=lambda x: x.distance)

        img_matches = cv2.drawMatches(new_gray_q, kp_q, new_gray_d, kp_d, matches, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # matchesThickness = 3, 

        if(self.radiusMatching == 0):
            src = np.float32([kp_q[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
            dst = np.float32([kp_d[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
            
            F, mask = cv2.findFundamentalMat(src, dst, cv2.RANSAC, 5.0)
            
            E = self.K_d.T @ F @ self.K
            
            _, R, t, mask = cv2.recoverPose(E, src, dst) # , focal=focal_, pp=pp_

            
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

            
            return img_matches, matches, R, t, E_update, F_empty
        else:
            new_kp_q = []
            new_kp_d = []

            for k in range(0, len(matches)):
                
                diff_x = kp_d[matches[k].trainIdx].pt[0] - kp_q[matches[k].queryIdx].pt[0]
                diff_y = kp_d[matches[k].trainIdx].pt[1] - kp_q[matches[k].queryIdx].pt[1]
                
                if((diff_x**2 + diff_y**2)**(1/2) <= self.radius):
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
                
            # Draw the matches
            new_img_matches = cv2.drawMatches(new_gray_q, new_kp_q, new_gray_d, new_kp_d, new_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) # matchesThickness = 3, 
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
                
            src = np.array([new_kp_q[match.queryIdx].pt for match in new_matches]).reshape(-1, 1, 2)
            dst = np.array([new_kp_d[match.trainIdx].pt for match in new_matches]).reshape(-1, 1, 2)
            
            # Compute the Fundamental matrix using RANSAC
            F, mask = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC, 5.0)
            
            # if(F == None):
            #     F_empty = 1
            #     return new_img_matches, new_matches, 0, 0, E_update, F_empty
            
            
            E = self.K.T @ F @ self.K_d

            # Normalize the Essential matrix
            U, S, Vt = np.linalg.svd(E)
            S_new = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
            E_new = U @ S_new @ Vt
            
            E_update = np.dot(E_update,E_new)
            _, R, t, mask = cv2.recoverPose(E_update, src, dst)
            
            # print('R: \n', R)
            # print('t: \n', t)
            # exit(0)

            return new_img_matches, new_matches, R, t, E_update, F_empty

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