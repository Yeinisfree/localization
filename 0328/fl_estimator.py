import cv2
import numpy as np
import time

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

def K_optim (src, src_prev, K, focal_factors) :
    largest = 0
    largest_f = 0
    
    for one in focal_factors:
        test_K = K.copy()
        test_K[0][0] = K[0][0] * one
        test_K[1][1] = K[1][1] * one
        
        E, inlier_mask= cv2.findEssentialMat(src, src_prev, test_K, method=cv2.RANSAC, prob = 0.999, threshold = 0.5)
        
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


# K = np.array([[301.39596558, 0.0, 316.70672662],
#                             [0.0, 300.95941162, 251.54445701],
#                             [0.0, 0.0, 1.0]])

# img1 = cv2.imread('/home/cgv/0223/dataset/Part2_undistort/undistort_5.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('/home/cgv/0223/dataset/Part2_undistort/undistort_6.jpg', cv2.IMREAD_GRAYSCALE)

# print("- Input K:")
# print(K)
# print()

# start = time.time()
# new_K = FL_estimator(K, img1, img2)
# end = time.time()
# print(f"- time: {end - start: .5f} sec")

# print("\n- Output new K:")
# print(new_K)