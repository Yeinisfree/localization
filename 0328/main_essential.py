import matplotlib.pyplot as plt
import time
import RIST 

if __name__ == '__main__':

    # main 객체 생성
    rist = RIST.RIST()

    #MAKE DB MAP
    rist.cm.db_map()

    ############## Main Loop ##############
    for i in range(1, rist.get_answer()+1):

        t_start = time.time()

        #################### Tracking Part ###################
        # if(rist.__focal_optim == 1):
        #     focal = int(rist.rm.K_list[i][0][0] + rist.rm.K_list[i][1][1])/2
        #     pp = (int(rist.rm.K_list[i][0][2]), int(rist.rm.K_list[i][1][2]))
        # else:
        #     focal = int(rist.rm.K[0][0] + rist.rm.K[1][1])/2
        #     pp = (int(rist.rm.K[0][2]), int(rist.rm.K[1][2]))
        
        # focal = int(rist.rm.K_list[i][0][0] + rist.rm.K_list[i][1][1])/2
        # pp = (int(rist.rm.K_list[i][0][2]), int(rist.rm.K_list[i][1][2]))
        
        # rist.tracking( i, focal, pp)
        rist.tracking(i)
        #################### Matching Part ####################
        # i = 7 # 특정 query에 대하여 볼 때
        rist.matching(i)
        # plt.show() # 특정 query에 대하여 볼 때
        # exit(0) # 특정 query에 대하여 볼 때

        t_end = time.time()
        print(f"{t_end - t_start:.5f} sec")

        plt.pause(0.1)

    plt.show()
    