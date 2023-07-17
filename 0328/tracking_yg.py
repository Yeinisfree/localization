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



def remove_outliers(prev_pts, next_pts, threshold):
    
    motion_vectors =  prev_pts - next_pts
    
    direction_vector = motion_vectors / np.linalg.norm(motion_vectors, axis=1, keepdims=True)
    
    # Depending on direction
    while True:
        mean_direction_vector = np.mean(direction_vector, axis=0)
        mean_direction_vector = mean_direction_vector / np.linalg.norm(mean_direction_vector)
        # print(mean_direction_vector)
        # A dot product of two vectors is a scalar value that represents the cosine similarity between the two vectors. 
        # If the dot product is positive, it means that the two vectors point in a similar direction. 
        # If the dot product is negative, it means that the two vectors point in opposite directions.
        # cosine similarity = dot(A, B) / (norm(A) * norm(B))
        dot_products = np.dot(direction_vector , mean_direction_vector.T)
        # print(dot_products)
        mask = dot_products > threshold
        outliers = np.where(mask == False) 
        if len(outliers[0]) == 0:
            break
        prev_pts = prev_pts[mask]
        next_pts = next_pts[mask] 
        direction_vector = direction_vector[mask]
        motion_vectors = motion_vectors[mask]
    
    # Depending on vectgor's length
    lengths = np.sqrt(np.sum(motion_vectors**2, axis=1))
    threshold = np.mean(lengths) + 2*np.std(lengths)
    mask = lengths < threshold
    prev_pts_filtered = prev_pts[mask]
    next_pts_filtered = next_pts[mask]
    
    
    
    
    return prev_pts_filtered, next_pts_filtered

def feature_detection(img, nfeatures):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures)
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray1, None)
    return kp, des
    

def feature_matching(kp1, kp2, desc1, desc2, matcher):
    
    if matcher == "BF":
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good_matches.append(m)
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        
        return src_pts, dst_pts, good_matches
        
        
    elif matcher == "FLANN":
        pass

# Gaussian Filter

def GaussianFilter(kernel_size, sigma, img):
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = np.outer(kernel, kernel.transpose())
    img = cv2.filter2D(img, -1, gaussian_kernel)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img

def calc_optical_flow(img1, img2):
    
    
    # Define the parameters for the Lucas-Kanade algorithm
    lk_params = dict(winSize=(3, 3), # 21 21
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, mask=None, qualityLevel=0.01, minDistance=12)
    curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    

    # # Gaussian Filtering
    # kernel_size = 5
    # sigma = 0.5
    

    
    # prev_gray = GaussianFilter(kernel_size, sigma, img1)    
    # prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=1000, mask=None, qualityLevel=0.01, minDistance=12, blockSize=3)
    # curr_gray = GaussianFilter(kernel_size, sigma, img2)
    # next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    
    src_pts = prev_pts[status == 1]
    dst_pts = next_pts[status == 1]
    return src_pts, dst_pts
    

def get_mean_direction_vector(src_pts, dst_pts):
    motion_vectors = src_pts - dst_pts
    direction_vector = motion_vectors / np.linalg.norm(motion_vectors, axis=1, keepdims=True)
    mean_direction_vector = np.mean(direction_vector, axis=0)
    mean_direction_vector = mean_direction_vector / np.linalg.norm(mean_direction_vector)
    
    return mean_direction_vector

def get_angle(direction): # mean_direction_vector
    reference = (1, 0)
    dot_product = direction[0] * reference[0] + direction[1] * reference[1]
    angle = math.acos(dot_product)
    angle =  angle * 180 / math.pi
    return angle


def get_Rt(src_pts, dst_pts, K, RT_update):

    # # Compute the Fundamental matrix using RANSAC
    # F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 3, 0.999)
    # # Compute the Essential matrix
    # E = np.dot(np.dot(K.T, F), K)

    # # Normalize the Essential matrix
    # U, S, Vt = np.linalg.svd(E)
    # S_new = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    # E_new = U @ S_new @ Vt
        
    
    
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)
    
    RT = makeRTMatrix(R, t.ravel())
    
    RT_update = RT_update.dot(RT)
    
    R_, t_ = recoverRTMatrix(RT_update)
     
    return R_, t_, RT_update

def detect_rotation(prev_rot_flag, flags, angle, threshold, direction, mean_direction_vector):
    
    if angle > threshold:
        curr_rot_flag = True
    else:
        curr_rot_flag = False
    
    if prev_rot_flag == False and curr_rot_flag == True:
        if flags == 0: 
            flags += 1
            direction = get_direction(mean_direction_vector, direction)
            # print(mean_direction_vector)
        else: 
            flags -= 1
            direction = get_direction(mean_direction_vector, direction)
           # print(mean_direction_vector)
    
    return curr_rot_flag, flags, direction
    
def get_direction(mean_direction_vector, direction):
    if mean_direction_vector[0] > 0:
        direction[0] = 1
    else:
        direction[0] = -1
    if mean_direction_vector[1] > 0:
        direction[1] = 1
    else:
        direction[1] = -1
        
    return direction
    
def get_translation(t, translation_xy, flags, direction):
       
    x_translation = t[0][0]
    y_translation = t[1][0] 
    
    # x_translation = t[0][0] /t[2][0]
    # y_translation = t[1][0] /t[2][0]
    
    # x_translation = abs(t[0][0])
    # y_translation = abs(t[1][0])
       
    # x_translation = abs(t[0][0] / t[2][0])
    # y_translation = abs(t[1][0] / t[2][0])
    
    # translation_xy[0] += x_translation 
    # translation_xy[1] += y_translation 
    
    
    if flags == 0:
        translation_xy[0] += x_translation * direction[1]
        translation_xy[1] += y_translation * direction[0]
    else:
        translation_xy[0] += y_translation * direction[0]
        translation_xy[1] += x_translation * direction[1]
        
    return translation_xy

