import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv
import tracking_yg_0330 as track


dataset = int(input("dataset 1/19 = 0, 3/17 = 1, AF = 2: "))
part = int(input("Part? "))

num_images = 0
file_dir = ""
ratio = 0

if dataset == 0:
    
    if part == 1:
        num_images = 51
        file_dir = "/home/aaron/study/data/Part1_undistort/undistort_"
        ratio = 0.9
    elif part == 2:
        num_images = 55
        file_dir = "/home/aaron/study/data/Part2_undistort/undistort_"
        ratio = 0.75
    else:
        num_images = 92
        file_dir = "/home/aaron/study/data/Part3_undistort/undistort_"
        ratio = 0.85
    
elif dataset == 1:
    if part == 1:
        num_images = 39
        file_dir = "/home/aaron/study/data/230317_test/part1_undistort/undistort_"
        ratio = 0.9
    elif part == 2:
        num_images = 44
        file_dir = "/home/aaron/study/data/230317_test/part2_undistort/undistort_"
        ratio = 0.75
    else:
        num_images = 85
        file_dir = "/home/aaron/study/data/230317_test/part3_undistort/undistort_"
        ratio = 0.85

else:
    if part == 2:
        num_images = 55
        file_dir = "/home/aaron/study/data/part2_AF_revise/test"
        ratio = 0.75
    elif part == 3:
        num_images = 78
        file_dir = "/home/aaron/study/data/part3_AF_revise/test"
        ratio = 0.75
        
        

images = []

for i in range(0, num_images):
    img = cv2.imread(file_dir +  str(i) + ".jpg")
    images.append(img)

# undistortion images camera intrinsic parameter
K = np.array([[301.39596558, 0.0, 316.70672662],
                         [0.0, 300.95941162, 251.54445701],
                         [0.0, 0.0, 1.0]])


translation_xy = [0.0, 0.0]
RT_update = np.eye(4)

nfeatures = 1000

angle_th = 30

flags = 1


for i in range(1, num_images):
    
    kp1, des1 = track.feature_detection(images[i-1], nfeatures)
    kp2, des2 = track.feature_detection(images[i], nfeatures)
    
    src_pts, dst_pts, good_matches = track.feature_matching(kp1, kp2, des1, des2, "BF", ratio)
    
    
    # src_pts, dst_pts, mean_direction_vector = track.remove_outliers_direction_manual(src_pts, dst_pts, rmv_th)
    src_pts, dst_pts, mean_direction_vector = track.remove_outliers_direction_auto(src_pts, dst_pts)
    src_pts, dst_pts = track.remove_outliers_length(src_pts, dst_pts)

    # Method 1

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    angle = math.atan2(M[0,1], M[0,0]) * 180 / math.pi
    print(i, angle)
    if abs(angle) > angle_th:
        print("The rotation is suspected.")
        
   

    
    # Pose estimation
    
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts)
    
     
    RT = track.makeRTMatrix(R, t.ravel())
    
    RT_update = RT_update.dot(RT)
    
    R, t = track.recoverRTMatrix(RT_update)
    
    translation = t
    
    if flags == 0:    
        translation_xy[0] += translation[0][0] 
        translation_xy[1] += translation[1][0] 
    else:
        translation_xy[0] += translation[1][0] 
        translation_xy[1] += translation[0][0] 
    
 

    template = np.zeros_like(images[i-1])
    curr_img = images[i].copy()
    prev_img = images[i-1].copy()
        
    # Draw the tracks on the mask image
    for j, (new, old) in enumerate(zip(dst_pts, src_pts)):
        a, b = new.ravel()
        c, d = old.ravel()
        template = cv2.line(template, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        curr_img = cv2.circle(curr_img, (int(a), int(b)), 5, (0, 0, 255), -1)
        prev_img = cv2.circle(prev_img, (int(c), int(d)), 5, (255, 0, 0), -1)
    template = cv2.line(template, (200, 200), (int(200+mean_direction_vector[0]*100), int(200+mean_direction_vector[1]*100)), (255, 255, 255), thickness=4)
    # Show the result
    dst = cv2.addWeighted(prev_img, 0.5, curr_img, 0.5, 0)
    
    res = cv2.add(dst, template)
    res = cv2.resize(res, (1280, 720))
    cv2.imwrite('./img_match/viz' + str(i) +'.jpg', res)
    
    
    
    
    plt.scatter(translation_xy[0], translation_xy[1])
    plt.annotate(str(i),xy=(translation_xy[0],translation_xy[1]), xytext=(translation_xy[0]+0.05,translation_xy[1]+0.05))
plt.show()