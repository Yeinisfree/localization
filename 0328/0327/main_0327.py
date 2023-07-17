import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

import tracking_0327 as track

# old
# part2 num = 56,  rmvoutlier_th:0.6  


# new
# part2 num = 44,  rmvoutlier_th:0.8

### 0317 dataset
## part 1
    # num_images = 40
    # rmv_th = 0.6    # 0.5 0.6 0.7 0.8 0.9
    # angle_th = 60
## part 2
    # num_images = 44
    # rmv_th = 0.8    # 0.5 0.6 0.7 0.8 0.9
    # angle_th = 110
## part 3
    # num_images = 86
    # rmv_th = x    # 0.5 0.6 0.7 0.8 0.9
    # angle_th = x

num_images = 92

rmv_th = 0.9
angle_th = 150 #110  
flags = 1

images = []

for i in range(0, num_images):
    img = cv2.imread("/home/cgv/0306_0325/dataset/Part3_undistort/undistort_"  + str(i) + ".jpg")
    # img = cv2.imread("/home/aaron/study/data/Part2_undistort/undistort_" + str(i) + ".jpg")
    images.append(img)

# undistortion images camera intrinsic parameter
K = np.array([[301.39596558, 0.0, 316.70672662],
                         [0.0, 300.95941162, 251.54445701],
                         [0.0, 0.0, 1.0]])


RT_update = np.eye(4)
translation_xy = [0.0, 0.0]

nfeatures = 1000

prev_rot_flag = False

direction = [1, 1]

for i in range(1, num_images):
    
    # ################# Method 1 #########################
    # src_pts, dst_pts = track.calc_optical_flow(images[i-1], images[i])
    
    
    # # #################### Method 2 #########################
    kp1, des1 = track.feature_detection(images[i-1], nfeatures)
    kp2, des2 = track.feature_detection(images[i], nfeatures)
    
    src_pts, dst_pts, good_matches = track.feature_matching(kp1, kp2, des1, des2, "BF")
    
    src_pts, dst_pts = track.remove_outliers_direction(src_pts, dst_pts, rmv_th)

    # src_pts, dst_pts = track.remove_outliers_length(src_pts, dst_pts)
    
    mean_direction_vector = track.get_mean_direction_vector(src_pts, dst_pts)
    
    angle = track.get_angle(mean_direction_vector)
    print(i, angle)
    
    R, t, RT_update = track.get_Rt(src_pts, dst_pts, K, RT_update)
    # print(t)
    curr_rot_flag, flags, direction = track.detect_rotation(prev_rot_flag, flags, angle, angle_th, direction, mean_direction_vector)
    
     
    translation_xy = track.get_translation(t, translation_xy, flags, direction)

    prev_rot_flag = curr_rot_flag
    
    
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
    cv2.imwrite('/home/cgv/img_match/viz' + str(i-1) +'.jpg', res)
    
    # print("translation_x: ", translation_xy[0], "translation_y: ", translation_xy[1])
    
    plt.scatter(translation_xy[0], translation_xy[1])
    plt.annotate(str(i),xy=(translation_xy[0],translation_xy[1]), xytext=(translation_xy[0]+0.05,translation_xy[1]+0.05))
plt.show()