import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import tracking
from collections import Counter

### setting ###
# DB Map
# 1. 촬영한 DB의 이미지들을 확인하여 section별 이미지의 수를 파악
# 2. 'save_DBs_pose' 안 '# section & db pose' 수정          quantization 끄고 하는 것 추천
# 3. decision_axis 함수 수정
#
# Query Tracking
# 4. tracking 궤적 확인
#       # recalib 중        if(prev_rot_suspectation != 1 or rot_suspectation != 1) and __ax != 4: --> 특정 section에서 recalib 끄고 궤적 확인
# 5. tracking.py --> get_translation_Essential 안 '이전 포즈 += translation' 파트에서 translation = t 부분 결과를 확인하며 점이 어디가는지 확인하고 수정
# 6. (4. ~ 5.) 궤적에 맞춰scale 수정 + '2.' 수정

__db_imgs_list = []
__DBs_pose = []
__cur_loc_db = [0, 0]

__section1 = 4
__section2 = 13

### 실험할 것 ###
__scale_x = 50 # DB 한 칸 한 칸 거리 조절, (0, 0) (0, 1) (0, 2) --> (0*__scale_x, 0*__scale_y) (0*__scale_x, 1*__scale_y) (0*__scale_x, 2*__scale_y)
__scale_y = 50

__range_x = 4.5 * __scale_x # 현재 위치 기준 앞 뒤로 몇 개의 db까지 매칭할 것인지, ex) 4.5 --> 현재 위치 기준 앞으로 4.5개 뒤로 4.5개 총 9개 db 검출
__range_y = 4.5 * __scale_y
###############
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
    ## section 1
    for i in range(0, __section1):
        __DBs_pose.append([0, i * __scale_y])
        
    section1_x = __DBs_pose[-1][0]
    section1_y = __DBs_pose[-1][1]
    
    section1_x /= __scale_x
    section1_y += 1 * __scale_y

    ## section 2
    for i in range(__section1, __section2):
        __DBs_pose.append([section1_x * __scale_x, section1_y])
        section1_x -= 1
        
    section2_x = __DBs_pose[-1][0]
    section2_y = __DBs_pose[-1][1]
    
    section2_y /= __scale_y
    section2_x -= 1 * __scale_x
        
    ## section 3
    for i in range(__section2, __num_db_imgs):
        __DBs_pose.append([section2_x, section2_y * __scale_y])
        section2_y += 1
    
    __num_db_imgs = len(__DBs_pose)
    
    # # visualization for checking - DB Map
    # # print(__DBs_pose)
    
    # for i in range(0, len(__DBs_pose)):
    #     plt.scatter(__DBs_pose[i][0], __DBs_pose[i][1], 100, color = 'blue')
    #     plt.annotate(i, xy = (__DBs_pose[i][0], __DBs_pose[i][1]))
    # print("DB Map Visualization")
    # plt.show()
    # exit(0)
    
    return __db_imgs_list, __db_imgs_list_name, __num_db_imgs, __DBs_pose


def Quantization(tracking_pose, __DBs_pose, __num_db_imgs, cur_loc_db, DB_idx_list, prev_DB_idx_list, cur_db_idx, __range_x, __range_y):
    # search DBs within range
    min_diff_x = 100.0
    min_diff_y = 100.0
    DB_idx_list = []
    
    ###################
    for i in range(0, len(__DBs_pose)): # range(0, len(__DBs_pose)): 전체 <--> prev_DB_idx_list: (이전 위치 주변 DB 기준) 현재 tracking에서 나온 pose로부터 가장 가까운 DB 위치 찾기
        if(abs(__DBs_pose[i][0] - tracking_pose[0]) <= min_diff_x
        and abs(__DBs_pose[i][1] - tracking_pose[1]) <= min_diff_y):
            min_diff_x = abs(__DBs_pose[i][0] - tracking_pose[0])
            min_diff_y = abs(__DBs_pose[i][1] - tracking_pose[1])
            cur_loc_db = __DBs_pose[i].copy()
            cur_db_idx = i
            
    # search range setting
    x_range_left = cur_loc_db[0] - __range_x
    x_range_right = cur_loc_db[0] + __range_x
    y_range_bottom = cur_loc_db[1] - __range_y
    y_range_top =  cur_loc_db[1] + __range_y
    
    # # visualization
    # downleft = (x_range_left, y_range_bottom)
    # rect = patches.Rectangle(downleft, 2*__range_x, 2*__range_y, facecolor = 'None', edgecolor = 'black' )
    # plt.gca().add_patch(rect)
    
    # 범위 안에 있는 DB 찾기
    for i in range(0, __num_db_imgs):
        if (x_range_left < __DBs_pose[i][0] and __DBs_pose[i][0] < x_range_right
            and y_range_bottom < __DBs_pose[i][1] and __DBs_pose[i][1] < y_range_top):
            DB_idx_list.append(i)
    if(len(DB_idx_list) == 0):
        print("**********************************")
        print("Currently, Searched DB list (X)")
        print("So, Previous list --> Current list")
        print("**********************************")
        DB_idx_list = prev_DB_idx_list
    ###################
    
    # ###################
    # # search range setting
    # x_range_left = tracking_pose[0] - __range_x
    # x_range_right = tracking_pose[0] + __range_x
    # y_range_bottom = tracking_pose[1] - __range_y
    # y_range_top =  tracking_pose[1] + __range_y
    
    # # # visualization
    # # downleft = (x_range_left, y_range_bottom)
    # # rect = patches.Rectangle(downleft, 2*__range_x, 2*__range_y, facecolor = 'None', edgecolor = 'black' )
    # # plt.gca().add_patch(rect)
    
    # # 범위 안에 있는 DB 찾기
    # for i in range(0, __num_db_imgs):
    #     if (x_range_left < __DBs_pose[i][0] and __DBs_pose[i][0] < x_range_right
    #         and y_range_bottom < __DBs_pose[i][1] and __DBs_pose[i][1] < y_range_top):
    #         DB_idx_list.append(i)
    
    # if(len(DB_idx_list) != 0):
    #     for i in DB_idx_list:
    #         if(abs(__DBs_pose[i][0] - tracking_pose[0]) <= min_diff_x
    #         and abs(__DBs_pose[i][1] - tracking_pose[1]) <= min_diff_y):
    #             min_diff_x = abs(__DBs_pose[i][0] - tracking_pose[0])
    #             min_diff_y = abs(__DBs_pose[i][1] - tracking_pose[1])
    #             cur_loc_db = __DBs_pose[i].copy()
    #             cur_db_idx = i
    # else:
    #     print("**********************************")
    #     print("Currently, Searched DB list (X)")
    #     print("So, Previous list --> Current list")
    #     print("**********************************")
    #     DB_idx_list = prev_DB_idx_list
    # ###################
                
    return cur_loc_db, DB_idx_list, cur_db_idx

def Quantization_move_direction(tracking_pose, __DBs_pose, __num_db_imgs, cur_loc_db, DB_idx_list, prev_DB_idx_list, cur_db_idx, __range_x, __range_y, move_direction, __ax):
    # search DBs within range
    min_diff_x = 100.0
    min_diff_y = 100.0
    DB_idx_list = []
    
    ###################
    for i in range(0, len(__DBs_pose)): # range(0, len(__DBs_pose)): 전체 <--> prev_DB_idx_list: (이전 위치 주변 DB 기준) 현재 tracking에서 나온 pose로부터 가장 가까운 DB 위치 찾기
        if(abs(__DBs_pose[i][0] - tracking_pose[0]) <= min_diff_x
        and abs(__DBs_pose[i][1] - tracking_pose[1]) <= min_diff_y):
            min_diff_x = abs(__DBs_pose[i][0] - tracking_pose[0])
            min_diff_y = abs(__DBs_pose[i][1] - tracking_pose[1])
            cur_loc_db = __DBs_pose[i].copy()
            cur_db_idx = i
            
    backward_range = 6
            
    # search range setting
    # if(__ax == 1):
    #     if(move_direction == 1):
    #         x_range_left = cur_loc_db[0] - __range_x
    #         x_range_right = cur_loc_db[0] + __range_x
    #         y_range_bottom = cur_loc_db[1] - backward_range * __scale_y
    #         y_range_top =  cur_loc_db[1] + __range_y
    #     else:
    #         x_range_left = cur_loc_db[0] - __range_x
    #         x_range_right = cur_loc_db[0] + __range_x
    #         y_range_bottom = cur_loc_db[1] - __range_y
    #         y_range_top =  cur_loc_db[1] + backward_range * __scale_y
    # elif(__ax == 2):
    #     if(move_direction == 1):
    #         x_range_left = cur_loc_db[0] - backward_range * __scale_x
    #         x_range_right = cur_loc_db[0] + __range_x
    #         y_range_bottom = cur_loc_db[1] - __range_y
    #         y_range_top =  cur_loc_db[1] + __range_y
    #     else:
    #         x_range_left = cur_loc_db[0] - __range_x
    #         x_range_right = cur_loc_db[0] + backward_range * __scale_x
    #         y_range_bottom = cur_loc_db[1] - __range_y
    #         y_range_top =  cur_loc_db[1] + __range_y
    # elif(__ax == 3):
    #     if(move_direction == 1):
    #         x_range_left = cur_loc_db[0] - __range_x
    #         x_range_right = cur_loc_db[0] + __range_x
    #         y_range_bottom = cur_loc_db[1] - __range_y
    #         y_range_top =  cur_loc_db[1] + backward_range * __scale_y
    #     else:
    #         x_range_left = cur_loc_db[0] - __range_x
    #         x_range_right = cur_loc_db[0] + __range_x
    #         y_range_bottom = cur_loc_db[1] - backward_range * __scale_y
    #         y_range_top =  cur_loc_db[1] + __range_y
    # elif(__ax == 4):
    #     if(move_direction == 1):
    #         x_range_left = cur_loc_db[0] - __range_x
    #         x_range_right = cur_loc_db[0] + backward_range * __scale_x
    #         y_range_bottom = cur_loc_db[1] - __range_y
    #         y_range_top =  cur_loc_db[1] + __range_y
    #     else:
    #         x_range_left = cur_loc_db[0] - backward_range * __scale_x
    #         x_range_right = cur_loc_db[0] + __range_x
    #         y_range_bottom = cur_loc_db[1] - __range_y
    #         y_range_top =  cur_loc_db[1] + __range_y
    
    if(__ax == 1):
        if(move_direction == 1):
            x_range_left = cur_loc_db[0] - 1 * __scale_x
            x_range_right = cur_loc_db[0] + 1 * __scale_x
            y_range_bottom = cur_loc_db[1] - backward_range * __scale_y
            y_range_top =  cur_loc_db[1] + __range_y
        else:
            x_range_left = cur_loc_db[0] - 1 * __scale_x
            x_range_right = cur_loc_db[0] + 1 * __scale_x
            y_range_bottom = cur_loc_db[1] - __range_y
            y_range_top =  cur_loc_db[1] + backward_range * __scale_y
    elif(__ax == 2):
        if(move_direction == 1):
            x_range_left = cur_loc_db[0] - backward_range * __scale_x
            x_range_right = cur_loc_db[0] + __range_x
            y_range_bottom = cur_loc_db[1] - 1 * __scale_y
            y_range_top =  cur_loc_db[1] + 1 * __scale_y
        else:
            x_range_left = cur_loc_db[0] - __range_x
            x_range_right = cur_loc_db[0] + backward_range * __scale_x
            y_range_bottom = cur_loc_db[1] - 1 * __scale_y
            y_range_top =  cur_loc_db[1] + 1 * __scale_y
    elif(__ax == 3):
        if(move_direction == 1):
            x_range_left = cur_loc_db[0] - 1 * __scale_x
            x_range_right = cur_loc_db[0] + 1 * __scale_x
            y_range_bottom = cur_loc_db[1] - __range_y
            y_range_top =  cur_loc_db[1] + backward_range * __scale_y
        else:
            x_range_left = cur_loc_db[0] - 1 * __scale_x
            x_range_right = cur_loc_db[0] + 1 * __scale_x
            y_range_bottom = cur_loc_db[1] - backward_range * __scale_y
            y_range_top =  cur_loc_db[1] + __range_y
    elif(__ax == 4):
        if(move_direction == 1):
            x_range_left = cur_loc_db[0] - __range_x
            x_range_right = cur_loc_db[0] + backward_range * __scale_x
            y_range_bottom = cur_loc_db[1] - 1 * __scale_y
            y_range_top =  cur_loc_db[1] + 1 * __scale_y
        else:
            x_range_left = cur_loc_db[0] - backward_range * __scale_x
            x_range_right = cur_loc_db[0] + __range_x
            y_range_bottom = cur_loc_db[1] - 1 * __scale_y
            y_range_top =  cur_loc_db[1] + 1 * __scale_y
    
    # visualization
    downleft = (x_range_left, y_range_bottom)
    x = abs(x_range_left - x_range_right)
    y = abs(y_range_top - y_range_bottom)
    print("Down-Left: ", downleft)
    rect = patches.Rectangle(downleft, x, y, facecolor = 'None', edgecolor = 'black' )
    plt.gca().add_patch(rect)
    
    # 범위 안에 있는 DB 찾기
    for i in range(0, __num_db_imgs):
        if (x_range_left <= __DBs_pose[i][0] and __DBs_pose[i][0] <= x_range_right
            and y_range_bottom <= __DBs_pose[i][1] and __DBs_pose[i][1] <= y_range_top):
            DB_idx_list.append(i)
    if(len(DB_idx_list) == 0):
        print("**********************************")
        print("Currently, Searched DB list (X)")
        print("So, Previous list --> Current list")
        print("**********************************")
        DB_idx_list = prev_DB_idx_list
    
    return cur_loc_db, DB_idx_list, cur_db_idx

def recalib(__db_imgs_list, __DB_idx_list, __cur_db_idx, __DBs_pose, img_q, cur_loc_db, tracking_pose, prev_loc_db_idx, similar_loc_cnt, __db_imgs_list_name, dist_th = 400, count_th = 10, nndr = 0.7, nfeatures = 1000):
    sift = cv2.SIFT_create(nfeatures)
    bf = cv2.BFMatcher()
    num_prev_matches = 0
    best_idx = 0
    re = 0
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
                if(m.distance < nndr * n.distance):
                    temp_good_matches.append(m)
                    
        if(len(temp_good_matches) >= count_th):
            if(len(temp_good_matches) >= num_prev_matches):                
                num_prev_matches = len(temp_good_matches)
                best_idx = i
                __cur_loc_db = __DBs_pose[best_idx].copy()
                new_tracking_pose = __cur_loc_db
                cnt += 1
            
        print(f"DB: {__db_imgs_list_name[i]} - # good_matches: ", len(temp_good_matches))
    
    if(cnt > 0):
        re = 1
        print("***** Recalibration *****")
    else:
        re = 0
        __cur_loc_db = cur_loc_db
        new_tracking_pose = tracking_pose
        best_idx = __cur_db_idx


    if(best_idx == prev_loc_db_idx):
        similar_loc_cnt += 1
    else:
        similar_loc_cnt = 0
        
    if(similar_loc_cnt > 3):
        print("+++++ Similar Location +++++")
        range_const = 1.5
    else:
        range_const = 1
        
        
    return __cur_loc_db, best_idx, new_tracking_pose, re, range_const, similar_loc_cnt, len(temp_good_matches)     
    
def decision_axis(cur_db_idx, __section1, __section2, __section3, __ax):
    ## __ax == 0 --> x-axis
    ## __ax == 1 --> y-axis
    
    if(cur_db_idx <= __section1-1):
        __ax = 1
    elif(__section1 <= cur_db_idx and cur_db_idx <= __section2-1):
        __ax = 2
    elif(__section2 <= cur_db_idx and cur_db_idx <= __section3-1):
        __ax = 3
    else:
        __ax = 4
        
    return __ax


    

########## PATH ##########
db_path = "/media/cgv/새 볼륨/rist_dataset/0720_re/DB/" # "/home/cgv/0621/ver1/92_re_rot/"
qpath = "/media/cgv/새 볼륨/rist_dataset/0720_re/query_slow_0/" # "/media/cgv/새 볼륨/rist_dataset/6pp_test/230510/test" # "/media/cgv/새 볼륨/rist_dataset/6pp_test/230510/test" # "/home/cgv/upp/230410_part1_q/"
save_path = "/home/cgv/0720_5pp/matching_result/"
##########################


########## MAIN ##########
# Q 읽어옴
num_images = 63 # 40 #171 # 63
images = []

for i in range(0, num_images):
    img = cv2.imread(qpath + str(i) + ".jpg")
    img = cv2.resize(img, (640, 480))
    images.append(img)

# DB 사진 저장, 좌표 저장
db_imgs_list, db_imgs_list_name, num_db_imgs, DBs_pose = save_DBs_pose(db_path, images[0].shape)

#
nfeatures = 1000

# tracking parameters
prev_idx = 0    # index of previous query image
angle_th = 50   # rotation b.w. queries > angle_th --> rotating
ratio = 0.9     # NNDR
RT_update = np.eye(4)   # RT accumulation

# recalib. params
prev_DB_idx_list = [0]      # indexes of surrounding DBs from the previous location
DB_idx_list = [0]           # indexes of surrounding DBs from the previous location
cur_db_idx = 0              # Quantized current location
cur_loc_db = [0, 0]         # coordinate currently located
prev_loc_db_idx = 0         # Quantized previous location
prev_ax = 1                 # previous position moved to the x-axis / y-axis
prev_rot_suspectation = 0   # rotation(just before) --> (O): 1
similar_loc_cnt = 0         # number of consecutive equal positions
range_const = 1             # similar_loc_cnt >= 3 --> search range increase
re = 0                      # recalib(O) --> 1, (X) --> 0 
indexes = []                # When rotating, matched DBs
indexes_matches = []        # Num of matching pairs
item_idx0 = []
item_idx1 = []
max_0 = 0
found = 0

# camera
K = np.array([[301.39596558, 0.0, 316.70672662],
                         [0.0, 300.95941162, 251.54445701],
                         [0.0, 0.0, 1.0]])
cam_pos = [0.0, 0.0]


# tracking
track = tracking.Tracking()

# feature detection for first
prev_kp, prev_des = track.feature_detection(images[prev_idx], nfeatures)

plt.figure(1)

org_range_x = __range_x
org_range_y = __range_y

# for d in range(0, len(DBs_pose)):
#     plt.annotate(d, xy = (DBs_pose[d][0], DBs_pose[d][1]))

for i in range(1, num_images):
    print("Query Num.:", i)
    
    plt.cla()
    
    # for d in range(0, len(DBs_pose)):
    #     plt.annotate(d, xy = (DBs_pose[d][0], DBs_pose[d][1]))

    # feature detection & matching
    curr_kp, curr_des = track.feature_detection(images[i], nfeatures)
    
    src_pts, dst_pts = track.feature_matching(prev_kp, prev_des, curr_kp, curr_des, ratio)
    
    # remove outliers
    src_pts, dst_pts = track.remove_outliers_direction_auto(src_pts, dst_pts)
    src_pts, dst_pts = track.remove_outliers_length(src_pts, dst_pts)
    
    mean_direction_vector = track.get_mean_direction_vector()
    
    track.detect_rotation(i, angle_th)
    
    # cam_pos = track.get_translation_Essential(cam_pos, src_pts, dst_pts, K, __ax)
    cam_pos, distance_mean = track.get_translation(cam_pos, src_pts, dst_pts, __ax)    
    move_direction = track.get_move_direction(distance_mean)
    
    rot_suspectation = track.get_suspectation()
    
    prev_idx = i
    prev_kp, prev_des = curr_kp, curr_des
    
    # ### Tracking 결과만 확인 ###
    # plt.scatter(cam_pos[0], cam_pos[1], 20, "gray", zorder = 2)
    # plt.annotate(f"{i}", xy=(cam_pos[0], cam_pos[1]))
    # for db in range(0, num_db_imgs):
    #     plt.scatter(DBs_pose[db][0], DBs_pose[db][1], 120, "blue", zorder = 1)
    
    # plt.savefig(save_path + f"Map_q{i}.png")
    # ##########################
    
    ### parameter - for recalib ###
    if(rot_suspectation == 1):
        nndr = 0.85
        nfeatures = 1000
        dist_th = 300
        count_th = 8 # 8 15
    else:
        nndr = 0.8
        nfeatures = 1000
        dist_th = 200
        count_th = 10 # 8 # 10 30
        
    if(__range_x > org_range_x or __range_y > org_range_y):
        nndr += 0.1
        dist_th += 0
        count_th -= 5
    ###############################
    
    tracking_result = cam_pos.copy()
    
    # Normalization to DB coordinates, display만을 위해서
    # cur_loc_db, DB_idx_list, cur_db_idx = Quantization(cam_pos.copy(), DBs_pose, num_db_imgs, cur_loc_db, DB_idx_list, prev_DB_idx_list, cur_db_idx, __range_x, __range_y)
    cur_loc_db, DB_idx_list, cur_db_idx = Quantization_move_direction(cam_pos.copy(), DBs_pose, num_db_imgs, cur_loc_db, DB_idx_list, prev_DB_idx_list, cur_db_idx, __range_x, __range_y, move_direction, __ax)
    
    # recalib
    if(prev_rot_suspectation != 1 or rot_suspectation != 1):
    # if(prev_rot_suspectation != 1 or rot_suspectation != 1) and __ax != 4:
        __range_x = org_range_x
        __range_y = org_range_y
        
        __cur_loc_db, cur_db_idx, cam_pos, re, range_const, similar_loc_cnt, num_matches = recalib(db_imgs_list, DB_idx_list, cur_db_idx, DBs_pose, images[i], cur_loc_db, cam_pos.copy(), prev_loc_db_idx, similar_loc_cnt, db_imgs_list_name, dist_th, count_th, nndr, nfeatures)

        # 현재 db idx를 append하여 저장, 매칭 수도 저장
        indexes.append(cur_db_idx)
        indexes_matches.append(num_matches)
        
        # 빈도 수에 따라 저장
        sorted_indexes = Counter(indexes) # 많은 순서대로
        
        # 빈도 수 같은 것이 둘 이상
        if(len(sorted_indexes.most_common(2)) >= 2 and # 저장된 것들이 2개 이상이고
           sorted_indexes.most_common(2)[0][1] == sorted_indexes.most_common(2)[1][1]): # 많이 나온 1등 2등의 빈도 수가 같다면 (idx가 아니라 빈도 수) (= 최소 두 개의 빈도 수가 같다 = (1등 빈도 수 = 2등 빈도 수 = 3등 빈도 수 ... ) )
            
            sorted_indexes_matches = sorted(indexes_matches, reverse = True)
            
            print(sorted_indexes)
            print(sorted_indexes_matches)
            
            for iter1 in range(0, len(indexes_matches)):
                item = sorted_indexes_matches[iter1] # 매칭 수 중 가장 큰 수 --> 두 번째 --> 세 번째 ...
                item_idx = indexes[indexes_matches.index(item)] # 이때의 index
                
                print("매칭 수 중 큰 수: ", item)
                print("이때 idx: ", item_idx)
                
                for iter2 in range(0, len(sorted_indexes.most_common(2))):
                    if(item_idx == sorted_indexes.most_common(2)[iter2][0]): # 매칭 수 중 가장 큰 수의 idx와 같은 idx가 같은 빈도 수를 가진 목록에 있는지
                        best_idx = item_idx
                        found = 1
                        
                if(found == 1):
                    found = 0
                    break
        else:
            best_idx = sorted_indexes.most_common(2)[0][0]
                    
        # 이동 (좌표 이동과 관련된 것들만)
        # if(prev_rot_suspectation == 0 and rot_suspectation == 1):
        
        print("best_idx = ", best_idx)
        
        cur_db_idx = best_idx
        __cur_loc_db = __DBs_pose[cur_db_idx].copy()
        cam_pos = __cur_loc_db
        
        # list 초기화
        indexes = []
        indexes_matches = []
        max_0 = 0
         
    else:
        re = 0
        
        # recalib. 변수명 바꾸고
        _, cur_db_idx_rot, _, _, _, _, num_matches_rot = recalib(db_imgs_list, DB_idx_list, cur_db_idx, DBs_pose, images[i], cur_loc_db, cam_pos.copy(), prev_loc_db_idx, similar_loc_cnt, db_imgs_list_name, dist_th, count_th, nndr, nfeatures)
        
        # 현재 db idx를 append하여 저장, 매칭 수도 저장
        indexes.append(cur_db_idx_rot)
        indexes_matches.append(num_matches_rot)
        
        print("< In Rotation >")
    
    # 여러번 같은 위치라고 추측되면, 다음엔 range를 계속 넓혀서
    __range_x = __range_x * range_const
    __range_y = __range_y * range_const
    
    prev_loc_db_idx = cur_db_idx

    # recalibration이 되도 축 확인
    if(re == 1 or rot_suspectation == 1):
        __ax = decision_axis(cur_db_idx, __section1, __section2, __section3, __ax)
        
    # 회전 탐지되면 rt 초기화
    if(prev_ax != __ax):
        print("***** RT initialization *****")
        track.initialize__RT_update()
        
    prev_rot_suspectation = rot_suspectation
    prev_ax = __ax
    prev_DB_idx_list = DB_idx_list
    
    print("current Quantized DB: ", cur_db_idx)
    print()
    
    ### visualization --> map ###
    if(re == 1):
        plt.scatter(__cur_loc_db[0], __cur_loc_db[1], 80, "red", zorder = 4)
    plt.scatter(cur_loc_db[0], cur_loc_db[1], 30, "yellow", zorder = 3) # only quan.
    plt.scatter(tracking_result[0], tracking_result[1], 20, "gray", zorder = 2) # tracking_result <-> cam_pose
    # plt.annotate(f"{i}", xy=(cam_pos[0], cam_pos[1]))
    for db in range(0, num_db_imgs):
        plt.scatter(DBs_pose[db][0], DBs_pose[db][1], 120, "blue", zorder = 1)
    
    plt.legend(("Quan. + adj.", "only Quan.", "tracking result", "DB"), loc = "lower right")
    
    # save results
    colorq = cv2.cvtColor(images[i].copy(), cv2.COLOR_BGR2GRAY)
    result = cv2.hconcat([colorq, db_imgs_list[cur_db_idx]]) # if q --> db ver 1, db_imgs_list[cur_db_idx]를 prev_db_img 로 바꾸기
    cv2.imwrite(save_path + f"q_{i}__db_" + db_imgs_list_name[cur_db_idx], result)
    if(re == 1):
        cv2.imwrite(save_path + f"recalib_q_{i}__db_" + db_imgs_list_name[cur_db_idx], result)
    plt.savefig(save_path + f"Map_q{i}.png")
    
    plt.pause(0.1)
    
    print()
    
print("-------------- end --------------") 
plt.show()