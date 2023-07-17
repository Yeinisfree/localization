import cv2
import numpy as np
import matplotlib.pyplot as plt


def FL_estimator(K, img1, img2) :
    samples = 30
    max_ratio = 5
    min_ratio = 0.2
    
    f = 0
    focal_factors = []
    fstep = 1.0 / samples
    fscale = max_ratio - min_ratio
    
    # Extract features from both images
    detector = cv2.ORB_create() # for speed
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # Match features between the two images
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    src_next = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    for i in range(0, samples+1) : 
        
        focal_factors.append(min_ratio + fscale * f * f)
        f += fstep
        
        if (f >= 1):
            break
        
    # find a factor with the most inliers & scaling
    new_K = K_optim(src, src_next, K, focal_factors)

    return new_K

def K_optim (src, src_next, K, focal_factors) :
    largest = 0
    largest_f = 0
    
    for one in focal_factors:
        test_K = K.copy()
        test_K[0][0] = K[0][0] * one
        test_K[1][1] = K[1][1] * one
        
        E, inlier_mask= cv2.findEssentialMat(src, src_next, test_K, method=cv2.RANSAC, prob = 0.999, threshold = 0.5)
        
        num_inliers = cv2.countNonZero(inlier_mask)
        
        if num_inliers > largest:
            largest = num_inliers
            new_K = test_K
            largest_f = one
            
    #         # What if the maximum number of liars is the same...?
    
    # print("- largest #inliers: ", largest)
    # print("- largest focal coeff: ", largest_f)
    # print()
    # print(f"- facal length x {largest_f}\n")
    
    return new_K

def pose_estimator(K, img1, img2, E_prev):
    sift = cv2.SIFT_create(1000)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    kp_q, new_des_q = sift.detectAndCompute(img1, None)
    kp_d, new_des_d = sift.detectAndCompute(img2, None)

    matches = bf.match(new_des_q, new_des_d)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # src = np.array([kp_q[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    # dst = np.array([kp_d[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    
    ###
    new_kp_q = []
    new_kp_d = []

    for k in range(0, len(matches)):
        
        diff_x = kp_d[matches[k].trainIdx].pt[0] - kp_q[matches[k].queryIdx].pt[0]
        diff_y = kp_d[matches[k].trainIdx].pt[1] - kp_q[matches[k].queryIdx].pt[1]
        
        if((diff_x**2 + diff_y**2)**(1/2) <= 30):
            temp_q = cv2.KeyPoint(kp_q[matches[k].queryIdx].pt[0], kp_q[matches[k].queryIdx].pt[1], kp_q[matches[k].queryIdx].size, kp_q[matches[k].queryIdx].angle, kp_q[matches[k].queryIdx].response, kp_q[matches[k].queryIdx].octave)
            temp_d = cv2.KeyPoint(kp_d[matches[k].trainIdx].pt[0], kp_d[matches[k].trainIdx].pt[1], kp_d[matches[k].trainIdx].size, kp_d[matches[k].trainIdx].angle, kp_d[matches[k].trainIdx].response, kp_d[matches[k].trainIdx].octave)
            new_kp_q.append(temp_q)
            new_kp_d.append(temp_d)

    new_kp_q = tuple(new_kp_q)
    new_kp_d = tuple(new_kp_d)

    new_des_q = sift.compute(img1, new_kp_q)
    new_des_d = sift.compute(img2, new_kp_d)

    new_des_q = np.array(new_des_q[1])
    new_des_d = np.array(new_des_d[1])

    new_matches = bf.match(new_des_q, new_des_d)

    new_matches = sorted(new_matches, key=lambda x: x.distance)
    
    src = np.array([new_kp_q[match.queryIdx].pt for match in new_matches]).reshape(-1, 1, 2)
    dst = np.array([new_kp_d[match.trainIdx].pt for match in new_matches]).reshape(-1, 1, 2)
    ###
    
    F, mask = cv2.findFundamentalMat(src, dst, cv2.RANSAC, 5.0)
    
    E = K.T @ F @ K
    
    E_prev = E @ E_prev
    
    _, R, t, mask = cv2.recoverPose(E_prev, src, dst)
    
    tMat = np.zeros(3)
    
    tMat[0] = t[0]/t[2]
    tMat[1] = t[1]/t[2]
    tMat[2] = t[2]/t[2]
    
    return tMat, E_prev

path = '/home/cgv/0306_JUNO/dataset/Part2_undistort/'
num_images = 10 ###

K = np.array([[301.39596558 , 0.0          , 316.70672662],
    [0.0          , 300.95941162 , 251.54445701],
    [0.0          , 0.0          , 1.0]])

query_imgs = []

for i in range(0, num_images):
    img = cv2.imread(path+'undistort_' + str(i) + ".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    query_imgs.append(img)

pose = [0, 0]

E_prev = np.identity(3)

plt.figure(1)
plt.scatter(pose[0], pose[1], 90, 'red')
plt.annotate("0", (pose[0], pose[1]))

for i in range(0, num_images-1):
    img1 = query_imgs[i]
    img2 = query_imgs[i+1]
    
    # K = FL_estimator(K, img1, img2)
    
    tMat, E_prev = pose_estimator(K, img1, img2, E_prev)
    
    pose[0] = pose[0] + tMat[0]
    pose[1] = pose[1] + tMat[1]
    
    print("pose: \n", pose)
    
    plt.scatter(pose[0], pose[1], 90, 'red')
    plt.annotate(f"{i+1}", (pose[0], pose[1]))
plt.show()