import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import tracking_0414_part2 as tracking 

__db_imgs_list_name = []
__db_imgs_list = []
__db_imgs = []
__num_db_imgs = 0

__DBs_pose = []
__cur_loc_db = [0, 0]
__cur_loc_q = [0, 0]
__DB_idx_list = []

__section1 = 7
__scale_x = 11
__scale_y = 11

__range_x = 2.5 * __scale_x
__range_y = 2.5 * __scale_y

__ax = 1

def save_DBs_pose(db_path, q_shape):
    __temp_db_imgs_list_name = os.listdir(db_path)
    
    __num_db_imgs = len(__temp_db_imgs_list_name)
    
    __db_imgs_list_name = np.zeros_like(__temp_db_imgs_list_name)
    
    for i in range(0, __num_db_imgs):
        splited_name = __temp_db_imgs_list_name[i].split('.')
        img_num = int(splited_name[0])
        
        __db_imgs_list_name[img_num] = __temp_db_imgs_list_name[i]
    
    # save db imgs
    for i in range(0, __num_db_imgs):
        img = cv2.imread(db_path + __db_imgs_list_name[i])
        img = cv2.resize(img, (q_shape[1], q_shape[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        __db_imgs_list.append(img)  
    
    # section & db pose
    for i in range(0, __section1):
        __DBs_pose.append([0, i * __scale_y])
        
        section1_x = 0
        section1_y = i * __scale_y

    for i in range(__section1, __num_db_imgs):
        __DBs_pose.append([section1_x * __scale_x, section1_y])
        section1_x += 1
    
    return __db_imgs_list, __db_imgs_list_name, __num_db_imgs, __DBs_pose


def Quantization(tracking_pose, __DBs_pose, __num_db_imgs, cur_loc_db, DB_idx_list, cur_db_idx, __range_x, __range_y):
     
    # search range setting
    x_range_left = tracking_pose[0] - __range_x
    x_range_right = tracking_pose[0] + __range_x
    y_range_bottom = tracking_pose[1] - __range_y
    y_range_top =  tracking_pose[1] + __range_y
    
    # # visualization
    # downleft = (x_range_left, y_range_bottom)
    # rect = patches.Rectangle(downleft, 2*__range_x, 2*__range_y, facecolor = 'None', edgecolor = 'black' )
    # plt.gca().add_patch(rect)
    
    # search DBs within range
    min_diff = 100.0
    min_diff_x = 100.0
    min_diff_y = 100.0
    cur_loc_q = []
    DB_idx_list = []

    for i in range(0, __num_db_imgs):
        if (x_range_left < __DBs_pose[i][0] and __DBs_pose[i][0] < x_range_right
            and y_range_bottom < __DBs_pose[i][1] and __DBs_pose[i][1] < y_range_top):
            DB_idx_list.append(i)
    
    if(len(DB_idx_list) != 0):
        for i in DB_idx_list:
            if(abs(__DBs_pose[i][0] - tracking_pose[0]) <= min_diff_x
            and abs(__DBs_pose[i][1] - tracking_pose[1]) <= min_diff_y):
                min_diff_x = abs(__DBs_pose[i][0] - tracking_pose[0])
                min_diff_y = abs(__DBs_pose[i][1] - tracking_pose[1])
                cur_loc_db = __DBs_pose[i].copy()
                cur_loc_q = tracking_pose.copy()
                cur_db_idx = i

    __cur_loc_db = cur_loc_db
    __cur_loc_q = cur_loc_q
    __DB_idx_list = DB_idx_list
                
    return cur_loc_db, DB_idx_list, cur_db_idx

def recalib(__db_imgs_list, __DB_idx_list, __cur_db_idx, __DBs_pose, img_q, cur_loc_db, tracking_pose, dist_th, count_th, nndr = 0.7, nfeatures = 1000):
    sift = cv2.SIFT_create(nfeatures)
    bf = cv2.BFMatcher()
    num_prev_matches = 0
    best_idx = 0
    cnt = 0
    
    img_q = cv2.cvtColor(img_q, cv2.COLOR_BGR2GRAY)
    
    print("Searched DB: ", __DB_idx_list)
    
    for i in __DB_idx_list:
        img_d = __db_imgs_list[i]
        
        ### how to check the similarity ###
        ### 1. using matching ###
        kp_d, des_d = sift.detectAndCompute(img_d, None)
        kp_q, des_q = sift.detectAndCompute(img_q, None)
    
        matches = bf.knnMatch(des_q, des_d, k=2)
        
        ## filtering - matched distances ##
        good_matches = []
        temp_good_matches = []
        
        for m, n in matches:
            if(m.distance < dist_th):
                if m.distance < nndr * n.distance:
                    temp_good_matches.append(m)
                    
        if(len(temp_good_matches) >= count_th):
            if(len(temp_good_matches) > num_prev_matches):                
                num_prev_matches = len(temp_good_matches)
                best_idx = i
                good_matches = temp_good_matches
                __cur_loc_db = __DBs_pose[best_idx].copy()
                new_tracking_pose = __cur_loc_db
                cnt += 1
            
        print("# DBs' good_matches: ", len(temp_good_matches))
    
    if(cnt > 0):
        re = 1
        print("***** Recalibration *****")
    else:
        re = 0
        __cur_loc_db = cur_loc_db
        new_tracking_pose = tracking_pose
        best_idx = __cur_db_idx


    # if(re == 0 and __cur_loc_db == cur_loc_db)
        
    
    
    return __cur_loc_db, best_idx, new_tracking_pose, re
    
def decision_axis(cur_db_idx, __section1, __ax):
    ## __ax == 0 --> x-axis
    ## __ax == 1 --> y-axis
    
    if(cur_db_idx <= __section1):
        __ax = 1
    else:
        __ax = 0
        
    return __ax
    
    

    
db_path = "/home/cgv/0328/dataset/db_Part2/"
qpath = "/home/cgv/0328/part2_AF_0328/test"



# Q 읽어옴

num_images = 55
images = []

for i in range(0, num_images):
    img = cv2.imread(qpath + str(i) + ".jpg")
    images.append(img)

# DB 사진 저장, 좌표 저장
db_imgs_list, db_imgs_list_name, num_db_imgs, DBs_pose = save_DBs_pose(db_path, images[0].shape)

prev_idx = 0

nfeatures = 1000
ratio = 0.8
angle_th = 30
RT_update = np.eye(4)
DB_idx_list = [0]
cur_loc_db = [0, 0]
cur_db_idx = 0

K = np.array([[301.39596558, 0.0, 316.70672662],
                         [0.0, 300.95941162, 251.54445701],
                         [0.0, 0.0, 1.0]])

cam_pos = [0.0, 0.0]


# tracking
track = tracking.Tracking()

# feature detection for first
prev_kp, prev_des = track.feature_detection(images[prev_idx], nfeatures)

plt.figure(1)


for i in range(0, num_db_imgs):
        plt.annotate(i, xy = (DBs_pose[i][0], DBs_pose[i][1]))

for i in range(1, num_images):
    print("Query Num.:", i)
    
    plt.cla()
    plt.xlim([-10, 350])
    plt.ylim([-10, 90])
    
    # feature detection & matching
    curr_kp, curr_des = track.feature_detection(images[i], nfeatures)
    
    src_pts, dst_pts = track.feature_matching(prev_kp, prev_des, curr_kp, curr_des, ratio)
    
    # remove outliers
    src_pts, dst_pts = track.remove_outliers_direction_auto(src_pts, dst_pts)
    src_pts, dst_pts = track.remove_outliers_length(src_pts, dst_pts)
    
    mean_direction_vector = track.get_mean_direction_vector()
    
    track.detect_rotation(i, angle_th)
    
    cam_pos = track.get_translation_Essential(cam_pos, src_pts, dst_pts, K, __ax)
    rot_suspectation = track.get_suspectation()
    
    prev_idx = i
    prev_kp, prev_des = curr_kp, curr_des
    
    
    ### parameter - for recalib ###
    if(rot_suspectation == 1):
        nndr = 0.8
        nfeatures = 1000
        dist_th = 400
        count_th = 8
    else:
        nndr = 0.7
        nfeatures = 1000
        dist_th = 300
        count_th = 20
    ###############################
    
    
    # Normalization to DB coordinates, display만을 위해서
    cur_loc_db, DB_idx_list, cur_db_idx = Quantization(cam_pos.copy(), DBs_pose, num_db_imgs, cur_loc_db, DB_idx_list, cur_db_idx, __range_x, __range_y)
    
    #recalib
    __cur_loc_db, cur_db_idx, cam_pos, re = recalib(db_imgs_list, DB_idx_list, cur_db_idx, DBs_pose, images[i], cur_loc_db, cam_pos.copy(), dist_th, count_th, nndr, nfeatures)

    # recalibration이 되도 축 확인
    if(re == 1):
        track.initialize__RT_update()
        __ax = decision_axis(cur_db_idx, __section1, __ax)
        
    # 회전 탐지되면 축 확인
    if(rot_suspectation == 1):
        __ax = decision_axis(cur_db_idx, __section1, __ax)
    
    print("current Quantized DB: ", cur_db_idx)
    print()
    
    
    
    # plt.scatter(cur_loc_db[0], cur_loc_db[1], 30, "yellow", zorder = 3) # only quan.
    plt.scatter(__cur_loc_db[0], __cur_loc_db[1], 80, "red", zorder = 4)
    plt.scatter(cam_pos[0], cam_pos[1], 20, "gray", zorder = 2) # 실제로는 안 보이게
    plt.annotate(f"{i}", xy=cam_pos)
    for i in range(0, num_db_imgs):
        plt.scatter(DBs_pose[i][0], DBs_pose[i][1], 120, "blue", zorder = 1)
    
    plt.legend(("only Quan. (adj. x)", "Quan. + adj.", "qqmatching -> pose", "DB"), loc = "lower right")
    
    plt.pause(1)
    # plt.close()
    
plt.show()