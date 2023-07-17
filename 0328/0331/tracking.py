import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def makeRTMatrix(R, t):
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

def recoverRTMatrix(RT):
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

def remove_outliers_length(prev_pts, next_pts):
    
    motion_vectors =  prev_pts - next_pts
    
     # Depending on vectgor's length
    lengths = np.sqrt(np.sum(motion_vectors**2, axis=1))
    threshold = np.mean(lengths) + np.std(lengths)
    mask = lengths < threshold
    if len(prev_pts[mask]) < 20 or len(next_pts[mask]) < 20:
        return prev_pts, next_pts
    prev_pts_filtered = prev_pts[mask]
    next_pts_filtered = next_pts[mask]
    
    return prev_pts_filtered, next_pts_filtered


def remove_outliers_direction_manual(prev_pts, next_pts, rmv_th):
    
    motion_vectors =  prev_pts - next_pts
    direction_vector = motion_vectors / np.linalg.norm(motion_vectors, axis=1, keepdims=True)
    
    
    while True:
        mean_direction_vector = np.mean(direction_vector, axis=0)
        mean_direction_vector = mean_direction_vector / np.linalg.norm(mean_direction_vector)

        # A dot product of two vectors is a scalar value that represents the cosine similarity between the two vectors. 
        # If the dot product is positive, it means that the two vectors point in a similar direction. 
        # If the dot product is negative, it means that the two vectors point in opposite directions.
        # cosine similarity = dot(A, B) / (norm(A) * norm(B))
        dot_products = np.dot(direction_vector , mean_direction_vector.T)
        #print(dot_products)
        mask = dot_products > rmv_th
        outliers = np.where(mask == False) 
        if len(outliers[0]) == 0:
            break
        prev_pts = prev_pts[mask]
        next_pts = next_pts[mask] 
        direction_vector = direction_vector[mask]
        motion_vectors = motion_vectors[mask]
    
    
    return prev_pts, next_pts, mean_direction_vector

def remove_outliers_direction_auto(prev_pts, next_pts, max_iterations=500):
    
    if len(prev_pts) < 10 or len(next_pts) < 10:
        print("Not enough points to estimate fundamental matrix")
        return prev_pts, next_pts

    prev_pts_copy = prev_pts.copy()
    next_pts_copy = next_pts.copy()
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
            prev_pts_copy = prev_pts.copy()
            next_pts_copy = next_pts.copy()
            motion_vectors = prev_pts_copy - next_pts_copy
            direction_vector = motion_vectors / np.linalg.norm(motion_vectors, axis=1, keepdims=True)
            continue

        prev_pts_copy = prev_pts_copy[mask]
        next_pts_copy = next_pts_copy[mask]
        direction_vector = direction_vector[mask]

        if len(prev_pts_copy) < 30 and len(next_pts_copy) < 30:
            threshold = bins[np.argmax(hist)]
            np.delete(bins, np.argmax(hist))
            prev_pts_copy = prev_pts.copy()
            next_pts_copy = next_pts.copy()
            motion_vectors = prev_pts_copy - next_pts_copy
            direction_vector = motion_vectors / np.linalg.norm(motion_vectors, axis=1, keepdims=True)
            continue


    if iteration >= max_iterations:
        # print("Exceeded maximum iterations without convergence.")
        pass

    # print(prev_pts_copy)
    # print(next_pts_copy)
    # print(mean_direction_vector)
    # exit(0)

    return prev_pts_copy, next_pts_copy, mean_direction_vector



def feature_detection(img, nfeatures):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures)
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray1, None)
    return kp, des
    

def feature_matching(img1, img2, kp1, kp2, desc1, desc2, matcher, th):
    
    if matcher == "BF":
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < th * n.distance:
                good_matches.append(m)
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        
        # new_img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        
        # plt.figure(123123)
        # plt.imshow(new_img_matches)
        # plt.show()
        
        print(img1.shape)
        exit(0)
        
        return src_pts, dst_pts, good_matches
        
        
    elif matcher == "FLANN":
        pass


def calc_optical_flow(img1, img2):
    
    
    # Define the parameters for the Lucas-Kanade algorithm
    lk_params = dict(winSize=(3, 3), # 21 21
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, mask=None, qualityLevel=0.01, minDistance=12)
    curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    
    src_pts = prev_pts[status == 1]
    dst_pts = next_pts[status == 1]
    return src_pts, dst_pts
    


def get_Rt(src_pts, dst_pts, K, RT_update):

    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)
     
    RT = makeRTMatrix(R, t.ravel())
    
    RT_update = RT_update.dot(RT)
    
    R, t = recoverRTMatrix(RT_update)
    
    
    
    return R, t, RT_update


