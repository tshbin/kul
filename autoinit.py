import matplotlib.image as mpimg
import numpy as np
from drawme import draw_lmks_on_original
from utils import read_dataset
from scipy.stats import norm
import os
import pickle
import cv2

roi_size = (25, 100)


def getcc(countur):
    M = cv2.moments(countur)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)

def scale_mask_back(mskfile, oofile, skip_rg):
    """
    Performs autoinitialization



    :param mskfile: path to mask file (outout of unet)
    :param oofile: path to original radiograph
    :param skip_rg: the radiograph we're predicting on, so we will skip it from creating PCA models
    :return: initialization parameters, like central point of the shape, height and angle of bounding rectangle
    """
    PCA_MODELS = create_pca_model(skip_rg)
    print('performing autoinitializartion')
    #### statistical info ###
    areat_u = 14894 - 2 * 5000
    areat_l = 11469 - 2 * 2226
    width_u = 97 + 50
    width_l = 72 + 15
    y_loc_max_u = 824 + 3 * 56
    y_loc_min_u = 824 - 3 * 56
    y_loc_max_l = 1108 + 3 * 52
    y_loc_min_l = 1108 - 3 * 52

    msk = mpimg.imread(mskfile)
    msk = msk[:,:,0]


    # ------------ first re-cut-----------------------#
    x_half = np.zeros((1024, 25))
    msk = np.hstack((x_half, msk, x_half))

    y_half = np.zeros((50, 562))
    msk = np.vstack((y_half, msk, y_half))

    # -------------- up scale -----------------------#
    originialimg = cv2.imread(oofile, 0)
    cutmeres = originialimg[200:originialimg.shape[0]-200, originialimg.shape[1]//2 - 240:originialimg.shape[1]//2 + 240]
    msk = cv2.resize(msk, (cutmeres.shape[1],cutmeres.shape[0]))

    # ---------------- second re-cut --------------#
    x_half = np.zeros((msk.shape[0], originialimg.shape[1]//2 - 240))
    msk = np.hstack((x_half, msk, x_half))
    y_half = np.zeros((200, msk.shape[1]))
    msk = np.vstack((y_half, msk, y_half))
    cv2.imwrite('processedmask.tif', msk)
    msk = cv2.imread('processedmask.tif', 0)
    imgrgb = cv2.cvtColor(originialimg, cv2.COLOR_GRAY2RGB)


    ##### join counturs that are just on top of each other
    areas = []
    widths = []
    center_x = []
    center_y = []
    ret, thresh = cv2.threshold(msk, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        areas.append(cv2.contourArea(contours[i]))
        x, y, w, h = cv2.boundingRect(contours[i])
        widths.append(w)

        M = cv2.moments(contours[i])
        try:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        except ZeroDivisionError:
            cx = 0
            cy = 0
        center_x.append(cx)
        center_y.append(cy)
    upper_vertical = []
    lower_vertical = []

    for did in range(len(contours)):
        if areas[did] > areat_u and y_loc_min_u < center_y[did] < y_loc_max_l:
            if abs(center_y[did] - 824) <= abs(center_y[did] - 1108):
#                print('upper incisior')
                upper_vertical.append((contours[did], (center_x[did], center_y[did])))
            else:
 #               print('lower incisior')
                lower_vertical.append((contours[did], (center_x[did], center_y[did])))

    upper_join = []
    lower_join = []
    upper_vertical = sorted(upper_vertical, key=lambda x: x[1][0])
    lower_vertical = sorted(lower_vertical, key=lambda x: x[1][0])

    for did in range(len(upper_vertical) - 1):
        if abs(upper_vertical[did][1][0] - upper_vertical[did+1][1][0]) < 50:
            upper_join.append((did, did+1))

    for did in range(len(lower_vertical) - 1):
        if abs(lower_vertical[did][1][0] - lower_vertical[did+1][1][0]) < 20:
            lower_join.append((did, did+1))




    for item in upper_join:
        cv2.line(msk, upper_vertical[item[0]][1], upper_vertical[item[1]][1], (255,255,255), 50)

    for item in lower_join:
        cv2.line(msk, lower_vertical[item[0]][1], lower_vertical[item[1]][1], (255,255,255), 25)




    notsplit = True
    newmsk = msk.copy()

    while notsplit:

        split_conturs = []
        ret, thresh = cv2.threshold(newmsk, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mskrgb = cv2.cvtColor(msk,cv2.COLOR_GRAY2RGB)
        areas = []
        widths = []
        center_y = []
        for i in range(len(contours)):
            areas.append(cv2.contourArea(contours[i]))
            x, y, w, h = cv2.boundingRect(contours[i])
            widths.append(w)

            M = cv2.moments(contours[i])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            except ZeroDivisionError:
                cx = 0
                cy = 0
            center_y.append(cy)

        for did in range(len(contours)):
            if areas[did] > areat_u and y_loc_min_u <center_y[did] < y_loc_max_l:
                if abs(center_y[did] - 824) <= abs(center_y[did] - 1108):
#                    print('upper incisior')
#                    print('current width: {}'.format(widths[did]))
                    if widths[did] > width_u:
                        split_conturs.append(did)
                    else:
                        pass
                else:
 #                   print('lower incisior')
 #                   print('current width: {}'.format(widths[did]))
                    if widths[did] > width_l:
                        split_conturs.append(did)
                    else:
                        #cv2.drawContours(imgrgb, [contours[did]], 0, (255,0 , 0), 3)
                        pass

    # draw_lmks_on_original([], imgrgb, [])
    # newmsk = msk.copy()

        #print('sc : {}'.format(split_conturs))
        if len(split_conturs) > 0:
            for did in split_conturs:
                newmsk = split_double_countur(contours[did], newmsk)
        else:
            notsplit = False

    upper_counturs = []
    lower_counturs = []

    ret, thresh = cv2.threshold(newmsk, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # mskrgb = cv2.cvtColor(msk,cv2.COLOR_GRAY2RGB)
    # cv2.drawContours(mskrgb, contours, -1, (0, 255, 0), 3)
    # draw_lmks_on_original([], mskrgb, [])
    areas = []
    widths = []
    center_y = []
    for i in range(len(contours)):
        areas.append(cv2.contourArea(contours[i]))
        x, y, w, h = cv2.boundingRect(contours[i])
        widths.append(w)
        M = cv2.moments(contours[i])
        try:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        except ZeroDivisionError:
            cx = 0
            cy = 0
        center_y.append(cy)
    upper_areas = []
    lower_areas = []
    for did in range(len(contours)):
        if areas[did] > areat_u and y_loc_min_u < center_y[did] < y_loc_max_l:
            if abs(center_y[did] - 824) <= abs(center_y[did] - 1108):
  #              print('upper incisior')
                upper_counturs.append(contours[did])
                upper_areas.append((areas[did], center_y[did]))
            else:
   #             print('lower incisior')
                lower_counturs.append(contours[did])
                lower_areas.append((areas[did], center_y[did]))
    #together = lower_counturs + upper_counturs
    halfxoriginal = originialimg.shape[1]//2
    centers_l = []
    centers_u = []
    for dc in lower_counturs:
        #cv2.circle(imgrgb, getcc(dc), 10, (0,0,255), -1)
        centers_l.append(getcc(dc))
    for dc in upper_counturs:
        #cv2.circle(imgrgb, getcc(dc), 10, (0,0,255), -1)
        centers_u.append(getcc(dc))


    distances_u = []
    distances_l = []
    for idxcu in range(len(centers_u)):
    #    print(centers_u[idxcu])
        distances_u.append((centers_u[idxcu][0] - halfxoriginal, centers_u[idxcu], idxcu))
    for idxcu in range(len(centers_l)):
        distances_l.append((centers_l[idxcu][0] - halfxoriginal, centers_l[idxcu], idxcu))

    distances_u = sorted(distances_u, key=lambda x: x[0])
    distances_l = sorted(distances_l, key=lambda x: x[0])
    #print('distances_u : {}'.format(distances_u))
    #print('distances_l : {}'.format(distances_l))




    width_u_lim = 97 * 1.5
    width_l_lim = 72 * 1.5
    missing_u_tooth = []
    missing_l_tooth = []


    #insert if missing in between
    for i in range(len(distances_u) - 1):
        if abs(distances_u[i][1][0] - distances_u[i+1][1][0]) > width_u_lim:
            missing_u_tooth.append((i,i+1))

    for i in range(len(distances_l) - 1):
        if abs(distances_l[i][1][0] - distances_l[i+1][1][0]) > width_l_lim:
            missing_l_tooth.append((i,i+1))
    for missing_u_item in missing_u_tooth:
        tmpcenter_x = min(distances_u[missing_u_item[0]][1][0], distances_u[missing_u_item[1]][1][0]) + abs(distances_u[missing_u_item[0]][1][0] - distances_u[missing_u_item[1]][1][0])//2
        all_y = [x[1][1] for x in distances_u]
        tmpcenter_y = sum(all_y)//len(all_y)
        #cv2.circle(imgrgb, (tmpcenter_x, tmpcenter_y), 10, (255,0,0), -1) #remove me
        distances_u.append((tmpcenter_x - halfxoriginal, (tmpcenter_x, tmpcenter_y), None))

    for missing_l_item in missing_l_tooth:
        tmpcenter_x = min(distances_l[missing_l_item[0]][1][0], distances_l[missing_l_item[1]][1][0]) + abs(
            distances_l[missing_l_item[0]][1][0] - distances_l[missing_l_item[1]][1][0]) // 2
        all_y = [x[1][1] for x in distances_l]
        tmpcenter_y = sum(all_y) // len(all_y)
        distances_l.append((tmpcenter_x - halfxoriginal, (tmpcenter_x, tmpcenter_y), None))

    distances_u = sorted(distances_u, key=lambda x: abs(x[0]))
    distances_l = sorted(distances_l, key=lambda x: abs(x[0]))
#    print('distances_u : {}'.format(distances_u))
#    print('distances_l : {}'.format(distances_l))
    distances_u = distances_u[:4]
    distances_l = distances_l[:4]

    # print('distances_u : {}'.format(distances_u))
    # print('distances_l : {}'.format(distances_l))

    distances_u = sorted(distances_u, key=lambda x: x[1][0])
    distances_l = sorted(distances_l, key=lambda x: x[1][0])
    # print('distances_u : {}'.format(distances_u))
    # print('distances_l : {}'.format(distances_l))



    #insert from the sides if less than 4 for upper/lower
    while len(distances_u) < 4:
        #idxtopick = 0 if abs(distances_u[0][0]) < abs(distances_u[-1][0]) else -1
        idxtopick = -1
        mincenter = distances_u[idxtopick][1]

        if True: #distances_u[idxtopick][0] >= 0: # tooth #4
            tmpcenter_x = mincenter[0] + 97
            all_y = [x[1][1] for x in distances_u]
            tmpcenter_y = sum(all_y) // len(all_y)

            square_y = tmpcenter_y - 272//2
            square_x = tmpcenter_x - 98//2
            dsqare = originialimg[square_y:square_y + 272, square_x:square_x + 98]
            #dsqare = cv2.cvtColor(dsqare, cv2.COLOR_BGR2GRAY)
            dsqare = cv2.resize(dsqare, roi_size)
            dmse = score_mse(dsqare, PCA_MODELS[4])
           # print('tooth #4 mse: {}'.format(dmse))

            t1 = {'disatance': (tmpcenter_x - halfxoriginal, (tmpcenter_x, tmpcenter_y), None), 'mse': dmse[0]}
           # print(t1)
        idxtopick = 0
        mincenter = distances_u[idxtopick][1]
        if True: #distances_u[idxtopick][0] <=0:
#        else: # tooth #1
            tmpcenter_x = mincenter[0] - 97
            all_y = [x[1][1] for x in distances_u]
            tmpcenter_y = sum(all_y) // len(all_y)

            square_y = tmpcenter_y - 272//2
            square_x = tmpcenter_x - 98//2
            dsqare = originialimg[square_y:square_y + 272, square_x:square_x + 98]
            #dsqare = cv2.cvtColor(dsqare, cv2.COLOR_BGR2GRAY)
            dsqare = cv2.resize(dsqare, roi_size)
            dmse = score_mse(dsqare, PCA_MODELS[1])
           # print('tooth #1 mse: {}'.format(dmse))

            t2 = {'disatance': (tmpcenter_x - halfxoriginal, (tmpcenter_x, tmpcenter_y), None), 'mse': dmse[0]}
           # print(t2)

        if t1['mse'] < t2['mse']:
            distances_u.append(t1['disatance'])
        else:
            distances_u.append(t2['disatance'])
        distances_u = sorted(distances_u, key=lambda x: x[1][0])

    while len(distances_l) < 4:
        idxtopick = -1
        mincenter = distances_l[idxtopick][1]
        if True: #distances_l[idxtopick][0] >= 0: # tooth #8
            tmpcenter_x = mincenter[0] + 72
            all_y = [x[1][1] for x in distances_l]
            tmpcenter_y = sum(all_y) // len(all_y)


            square_y = tmpcenter_y - 250//2
            square_x = tmpcenter_x - 72//2
            dsqare = originialimg[square_y:square_y + 250, square_x:square_x + 72]
            #dsqare = cv2.cvtColor(dsqare, cv2.COLOR_BGR2GRAY)
            dsqare = cv2.resize(dsqare, roi_size)
            dmse = score_mse(dsqare, PCA_MODELS[5])
          #  print('tooth #5 mse: {}'.format(dmse))

            t1 = {'disatance': (tmpcenter_x - halfxoriginal, (tmpcenter_x, tmpcenter_y), None), 'mse': dmse}


        idxtopick = 0
        mincenter = distances_l[idxtopick][1]
        if True: #distances_l[idxtopick][0] <= 0:
        #else:  #tooth #5
            tmpcenter_x = mincenter[0] - 72
            all_y = [x[1][1] for x in distances_l]
            tmpcenter_y = sum(all_y) // len(all_y)


            square_y = tmpcenter_y - 250 // 2
            square_x = tmpcenter_x - 72 // 2
            dsqare = originialimg[square_y:square_y + 250, square_x:square_x + 72]
            # dsqare = cv2.cvtColor(dsqare, cv2.COLOR_BGR2GRAY)
            dsqare = cv2.resize(dsqare, roi_size)
            dmse = score_mse(dsqare, PCA_MODELS[8])
          #  print('tooth #8 mse: {}'.format(dmse))

            t2 = {'disatance': (tmpcenter_x - halfxoriginal, (tmpcenter_x, tmpcenter_y), None), 'mse': dmse}

        if t1['mse'] < t2['mse']:
            distances_l.append(t1['disatance'])
        else:
            distances_l.append(t2['disatance'])
        distances_l = sorted(distances_l, key=lambda x: x[1][0])







    upper_final = []
    lower_final = []
    distances_u = sorted(distances_u, key=lambda x: x[0])
    distances_l = sorted(distances_l, key=lambda x: x[0])

    # print('distances_u : {}'.format(distances_u))
    # print('distances_l : {}'.format(distances_l))

    #swap left/right if the distance is too big:
    if abs(distances_u[0][0]/distances_u[-1][0]) >= 1.7:

        tmpcenter_x = distances_u[0][1][0]
        tmpcenter_y = distances_u[0][1][1]


        square_y = tmpcenter_y - 272 // 2
        square_x = tmpcenter_x - 98 // 2
        dsqare = originialimg[square_y:square_y + 250, square_x:square_x + 72]
        dsqare = cv2.resize(dsqare, roi_size)
        dmse_l = score_mse(dsqare, PCA_MODELS[1])
        #print('tooth #1 mse: {}'.format(dmse_l))


        tmpcenter_x = distances_u[-1][1][0] + 97
        all_y = [x[1][1] for x in distances_u]
        tmpcenter_y = sum(all_y) // len(all_y)

        square_y = tmpcenter_y - 272 // 2
        square_x = tmpcenter_x - 98 // 2
        dsqare = originialimg[square_y:square_y + 250, square_x:square_x + 72]
        # dsqare = cv2.cvtColor(dsqare, cv2.COLOR_BGR2GRAY)
        dsqare = cv2.resize(dsqare, roi_size)
        dmse_r = score_mse(dsqare, PCA_MODELS[4])
        #print('tooth #4 mse: {}'.format(dmse_r))


        if dmse_r < dmse_l:
            distances_u.append((tmpcenter_x - halfxoriginal, (tmpcenter_x, tmpcenter_y), None))
            distances_u.pop(0)


    elif abs(distances_u[-1][0]/distances_u[0][0]) >= 1.7:
        tmpcenter_x = distances_u[-1][1][0]
        tmpcenter_y = distances_u[-1][1][1]

        square_y = tmpcenter_y - 272 // 2
        square_x = tmpcenter_x - 98 // 2
        dsqare = originialimg[square_y:square_y + 272, square_x:square_x + 98]
        dsqare = cv2.resize(dsqare, roi_size)
        dmse_r = score_mse(dsqare, PCA_MODELS[4])
        #print('tooth #4 mse: {}'.format(dmse_r))



        tmpcenter_x = distances_u[0][1][0] - 97
        all_y = [x[1][1] for x in distances_u]
        tmpcenter_y = sum(all_y) // len(all_y)


        square_y = tmpcenter_y - 272 // 2
        square_x = tmpcenter_x - 98 // 2
        dsqare = originialimg[square_y:square_y + 272, square_x:square_x + 98]
        dsqare = cv2.resize(dsqare, roi_size)
        dmse_l = score_mse(dsqare, PCA_MODELS[1])
        #print('tooth #1 mse: {}'.format(dmse_l))

        if dmse_l < dmse_r:
            distances_u.insert(0, (tmpcenter_x - halfxoriginal, (tmpcenter_x, tmpcenter_y), None))
            distances_u.pop(-1)

    if abs(distances_l[0][0]/distances_l[-1][0]) >= 2:

        tmpcenter_x = distances_l[0][1][0]
        tmpcenter_y = distances_l[0][1][1]


        square_y = tmpcenter_y - 250 // 2
        square_x = tmpcenter_x - 72 // 2
        dsqare = originialimg[square_y:square_y + 250, square_x:square_x + 72]
        dsqare = cv2.resize(dsqare, roi_size)
        dmse_l = score_mse(dsqare, PCA_MODELS[5])
        #print('tooth #1 mse: {}'.format(dmse_l))


        tmpcenter_x = distances_l[-1][1][0] + 72
        all_y = [x[1][1] for x in distances_l]
        tmpcenter_y = sum(all_y) // len(all_y)

        square_y = tmpcenter_y - 250 // 2
        square_x = tmpcenter_x - 72 // 2
        dsqare = originialimg[square_y:square_y + 250, square_x:square_x + 72]
        dsqare = cv2.resize(dsqare, roi_size)
        dmse_r = score_mse(dsqare, PCA_MODELS[8])
        #print('tooth #1 mse: {}'.format(dmse_r))


        if dmse_r < dmse_l:
            distances_l.append((tmpcenter_x - halfxoriginal, (tmpcenter_x, tmpcenter_y), None))
            distances_l.pop(0)


    elif abs(distances_l[-1][0]/distances_l[0][0]) >= 2:
        tmpcenter_x = distances_l[-1][1][0]
        tmpcenter_y = distances_l[-1][1][1]

        square_y = tmpcenter_y - 250 // 2
        square_x = tmpcenter_x - 72 // 2
        dsqare = originialimg[square_y:square_y + 250, square_x:square_x + 72]
        dsqare = cv2.resize(dsqare, roi_size)
        dmse_r = score_mse(dsqare, PCA_MODELS[8])
        #print('tooth #8 mse: {}'.format(dmse_r))



        tmpcenter_x = distances_l[0][1][0] - 72
        all_y = [x[1][1] for x in distances_l]
        tmpcenter_y = sum(all_y) // len(all_y)


        square_y = tmpcenter_y - 250 // 2
        square_x = tmpcenter_x - 72 // 2
        dsqare = originialimg[square_y:square_y + 250, square_x:square_x + 72]
        dsqare = cv2.resize(dsqare, roi_size)
        dmse_l = score_mse(dsqare, PCA_MODELS[5])
        #print('tooth #5 mse: {}'.format(dmse_l))

        if dmse_l < dmse_r:
            distances_l.insert(0, (tmpcenter_x - halfxoriginal, (tmpcenter_x, tmpcenter_y), None))
            distances_l.pop(-1)

    # print('distances_u : {}'.format(distances_u))
    # print('distances_l : {}'.format(distances_l))


    #print('------------------- heights -----------------')
    for did in range(len(distances_u)):
        if distances_u[did][-1] is not None:
            x, y, w, h = cv2.boundingRect(upper_counturs[distances_u[did][-1]])
            clip = y + h - (271)
            #print(h)
            if h > 271 + 88:
                print('clipping by {}'.format(clip))
                upper_counturs[distances_u[did][-1]] = np.clip(upper_counturs[distances_u[did][-1]], np.array([-np.inf, clip]), None).astype(int)
            upper_final.append(upper_counturs[distances_u[did][-1]])

            cx, cy = getcc(upper_counturs[distances_u[did][-1]])
            cv2.circle(imgrgb, (cx, cy), 10, (255, 0, 0), -1)
            distances_u[did] = (cx - halfxoriginal, (cx, cy), distances_u[did][-1])
        else:
            cv2.circle(imgrgb, distances_u[did][-2], 5, (0,255,0), -1)


    for did in range(len(distances_u)):
        if distances_u[did][-1] is not None:
            cx, cy = distances_u[did][1]
            rect = cv2.minAreaRect(upper_counturs[distances_u[did][-1]])
            angle = rect[2]
            if rect[1][0] < rect[1][1]:
                angle = angle - 90
            x, y, w, h = cv2.boundingRect(upper_counturs[distances_u[did][-1]])
            distances_u[did] = ((cx, cy), h, upper_counturs[distances_u[did][-1]], angle, w)
            # sys.exit(20)
        else:
            cx, cy = distances_u[did][1]
            distances_u[did] = ((cx, cy), 271, None, None, 95)

    for did in range(len(distances_l)):
        if distances_l[did][-1] is not None:
            cx, cy = distances_l[did][1]
            rect = cv2.minAreaRect(lower_counturs[distances_l[did][-1]])
            angle = rect[2]
            if rect[1][0] < rect[1][1]:
                angle = angle - 90

            x, y, w, h = cv2.boundingRect(lower_counturs[distances_l[did][-1]])
            distances_l[did] = ((cx, cy), h, lower_counturs[distances_l[did][-1]], angle, w)
        else:
            cx, cy = distances_l[did][1]
            distances_l[did] = ((cx, cy), 250, None, None, 72)


    # print('distances_u : {}'.format(distances_u))
    # print('distances_l : {}'.format(distances_l))

    for item in distances_u + distances_l:
        if item[2] is not None:
            cv2.drawContours(imgrgb, [item[2]], 0, (255,0,0), 3)
        else:
            cv2.circle(imgrgb, item[0], 10, (255,0,0), -1)
    print('autoinitializartion is done')
    return distances_u, distances_l





def process_mask(mskfile, orignialfile, oofile):
    msk = mpimg.imread(mskfile)
    msk = msk[:,:,0]
    print(msk.shape)
    ret, thresh = cv2.threshold(msk, 127, 255, 0)
    imgrgb = cv2.imread(orignialfile)
    print(imgrgb.shape)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    print('cs: {}'.format(contours[0].shape))
    print('cs: {}'.format(contours[0][0]))
    for i in range(len(contours)):
        areas.append(cv2.contourArea(contours[i]))
    areas = np.asarray(areas)
    max8idx = areas.argsort()[::-1]
    centrals = []
    for did in max8idx[:8]:
        cv2.drawContours(imgrgb, [contours[did]], 0, (0,255,0), 3)
        M = cv2.moments(contours[did])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centrals.append([cx, cy])
        #cv2.circle(imgrgb, (cx, cy), 3, (255 ,0 ,0), -1)
    #assign centrals to teeth:
    toothcentral = {1: None, 2: None, 3: None, 4: None,
                    5: None, 6: None, 7: None, 8: None}
    centrals_y_sorted = sorted(centrals, key=lambda x: x[1])
    print('cs: {}'.format(centrals_y_sorted))
    upper_teeth = sorted(centrals_y_sorted[:4], key=lambda x: x[0])
    lower_teeth = sorted(centrals_y_sorted[4:], key=lambda x: x[0])

    print('upper: {}'.format(upper_teeth))
    print('lower: {}'.format(lower_teeth))


    for did in range(len(upper_teeth)):
        toothcentral[5 + did] = upper_teeth[did]

    for did in range(len(lower_teeth)):
        toothcentral[1 + did] = lower_teeth[did]


    #rescale all back:


    origorig = cv2.imread(oofile)
    cutshape = [origorig.shape[0]-400, 480]
    xscalemul = cutshape[1]/562
    yscalemul = cutshape[0]/1124

    print(toothcentral)

    for did in toothcentral:
        toothcentral[did][0] = int((toothcentral[did][0] + 25) * xscalemul) + int(origorig.shape[1]/2 - 240)
        toothcentral[did][1] = int((toothcentral[did][1] + 50) * yscalemul) + 200

    cutmeimg = cv2.imread('/Users/bberlog/tmp/zzzz/cutme/X/01.tif')
    print(toothcentral)
    # print('after mul: {}'.format(toothcentral[1]))
    #
    # sys.exit(0)
    # for did in toothcentral:
    #     toothcentral[did][0] += xscalesum
    #     toothcentral[did][1] += yscalesum
    #     toothcentral[did][0] = int(toothcentral[did][0] * xscalemul)
    #     toothcentral[did][1] = int(toothcentral[did][1] * yscalemul)
    #

    #print(toothcentral[4])
    for did in toothcentral:
        cv2.circle(origorig, (int(toothcentral[did][0]), int(toothcentral[did][1])), 10, (255, 0, 0), -1)
    draw_lmks_on_original([], origorig, [])
    # cv2.imshow('cnt', origorig)
    # cv2.waitKey(0)


def get_stats(upper=True):
    """
    Used to identify statistical parameters for teeth, like width, height, central location.
    Used for debugging, currently not used.
    :param upper: if upper teeth
    :return: None
    """
    providedlmks = []
    lmk_location = '../Landmarks/original/'
    rg_location = '../Radiographs/'
    if upper:
        tooth = [1, 2, 3, 4]
    else:
        tooth = [5,6,7,8]
    for tn in tooth:
        dataset = read_dataset(tn, lmk_location, rg_location)
        for item in dataset:
            if item['rg']:
                providedlmks.append(item['lmk'])
                #img = item['image']
    #draw_lmks_on_original(providedlmks, img, [(255,0,0)] * 8)
    areas = []
    widths =[]
    centers_y = []
    heights = []
    for dlmk in providedlmks:
        rrr = np.expand_dims(dlmk, axis=1)
        rrr = rrr.astype(int)
        print(rrr.shape)
        print(rrr[0])
        M = cv2.moments(rrr)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centers_y.append(cy)

        darea = cv2.contourArea(rrr)
        x, y, w, h = cv2.boundingRect(rrr)
        widths.append(w)
        heights.append(h)
        areas.append(darea)
    results = {}
    #print(len(providedlmks))
    print('----------areas--------------')
    print(areas)
    param = norm.fit(areas)
    print('mean: {}'.format(param[0]))
    print('std: {}'.format(param[1]))
    results['area'] = {'mean': param[0], 'std': param[1]}
    print('----------widths--------------')
    print(widths)
    param = norm.fit(widths)
    print('mean: {}'.format(param[0]))
    print('std: {}'.format(param[1]))
    results['width'] = {'mean': param[0], 'std': param[1]}
    print('----------heights--------------')
    print(heights)
    param = norm.fit(heights)
    print('mean: {}'.format(param[0]))
    print('std: {}'.format(param[1]))
    results['height'] = {'mean': param[0], 'std': param[1]}
    print('----------centers--------------')
    print(centers_y)
    param = norm.fit(centers_y)
    print('mean: {}'.format(param[0]))
    print('std: {}'.format(param[1]))
    results['center_y'] = {'mean': param[0], 'std': param[1]}
    return results
# def identify_countours(stath=None, statw=None, statarea):

def split_double_countur(countur, msk, shift = 25):
    """
    Sometimes U-net mask fails to separate teeth by small margine, but that breaks our contur detection
    Hence, if the tooth is too wide, we perform a split of the mask and recalculate counturs


    :param countur: countur we need to split
    :param msk: origianl mask (numpy)
    :param shift: step to step from the side of countur
    :return: returns new msk (numpy array)
    """
     #shift how much step from sides of bounding box
    x, y, w, h = cv2.boundingRect(countur)
    onepiece = msk[y:y + h, x + shift:x + w - shift]
    longest_len = [(np.inf, [None, None], [None, None]) for _ in range(onepiece.shape[1])]
    insidewhite = False
    notfirst = False
    for dummyx in range(onepiece.shape[1]):
        if insidewhite:
            endpoint = [dummyy, dummyx - 1]
            longest_len[dummyx - 1] = (tmpl, startpoint, endpoint)
        elif not insidewhite and notfirst:
            longest_len[dummyx - 1] = (tmpl, startpoint, endpoint)
        tmpl = 0
        insidewhite = False
        notfirst = True
        for dummyy in range(onepiece.shape[0]):
            if onepiece[dummyy][dummyx] == 255 and not insidewhite:
                insidewhite = True
                tmpl += 1
                startpoint = [dummyy, dummyx]
            elif onepiece[dummyy][dummyx] == 255 and insidewhite:
                tmpl += 1
            elif onepiece[dummyy][dummyx] == 0 and insidewhite:
                insidewhite = False
                endpoint = [dummyy -1, dummyx]

    longest_len = sorted(longest_len, key=lambda x: x[0])
    linestart = longest_len[0][1]
    lineend = longest_len[0][2]
    linestart[0] += y
    lineend[0] += y
    linestart[1] += x + shift
    lineend[1] += x + shift
    mskrgb = cv2.cvtColor(msk,cv2.COLOR_GRAY2RGB)
    cv2.line(mskrgb, (linestart[1], linestart[0]), (lineend[1], lineend[0]), (0, 0, 0), 10)
    msk = cv2.cvtColor(mskrgb,cv2.COLOR_RGB2GRAY)
    return msk


def score_mse(img, model):
    img = img.flatten()
    img = np.reshape(img, (1, img.shape[0]))
    #print('img shape: {}'.format(img.shape))
    img_wo_mean = img - model[0]
    projected_images = np.dot(img_wo_mean, model[2])
    reconstructed_images = np.dot(projected_images, model[2].T) + model[0]
    mse = np.mean((img - reconstructed_images) **2, axis = 1)
    return mse


def create_pca_model(skip_rg):
    #global PCA_MODELS
    ttt = [1, 4, 5, 8]
    providedlmks = []
    lmk_location = '../Landmarks/original/'
    rg_location = '../Radiographs/'

    def pca(X, number_of_components):
        # centered matrix, max number_of_components = N - 1 , where N number of samples
        number_of_components = min(number_of_components, X.shape[0])
        # calcaulate mean and center input matrix
        mean = np.mean(X, axis=0)
        X_wo_mean = X - mean
        # calculate covariance matrix on centered
        covx = np.cov(X_wo_mean, rowvar=False)  # rows - items in dataset, columns are variables
        eigenvalues, eigenvectors = np.linalg.eig(covx)
        index_of_interest = np.argsort(eigenvalues)[-number_of_components:][::-1]
        eigenvalues = eigenvalues[index_of_interest]
        eigenvectors = eigenvectors[:, index_of_interest]
        eigenvalues = np.real(eigenvalues)  # due to computation issues, get rid of imagine part
        eigenvectors = np.real(eigenvectors)
        return [mean, eigenvalues, eigenvectors]



    if os.path.isfile('PCA_MODELS_{}.cpkl'.format(skip_rg)):
        print('Loading saved PCA model')
        with open('PCA_MODELS_{}.cpkl'.format(skip_rg), 'rb') as fr:
            PCA_MODELS = pickle.load(fr)
        return PCA_MODELS
    else:
        print('Did not find saved PCA model, will create a new one, it will take a while')
        PCA_MODELS = {}
        for tn in ttt:
            stackus = []
            dataset = read_dataset(tn, lmk_location, rg_location)
            for item in dataset:
                if item['rg'] != skip_rg:
                    img = item['image']
                    dlmk = item['lmk']
                    rrr = np.expand_dims(dlmk, axis=1)
                    rrr = rrr.astype(int)
                    x, y, w, h = cv2.boundingRect(rrr)
                    roi_color = img[y:y+h, x:x+w]
                    roi_resized = cv2.resize(roi_color, roi_size)
                    roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                    #print(roi_resized.shape)
                    tmp_n = roi_resized.flatten()
                    stackus.append(tmp_n)
            full_x = stackus[0]
            for did in range(1, len(stackus)):
                full_x = np.vstack((full_x, stackus[did]))

            #print('doing PCA')
            #print(full_x.shape)
            mean, eigenvalues, eigenvectors = pca(full_x,
                                                  number_of_components=9)  # 3 components seem to cover > 80% of variance
            PCA_MODELS[tn] = (mean, eigenvalues, eigenvectors)

        with open('PCA_MODELS_{}.cpkl'.format(skip_rg), 'wb') as fw:
            pickle.dump(PCA_MODELS, fw)
        return PCA_MODELS

if __name__ == '__main__':
    providedlmks = []
    lmk_location = '../Landmarks/original/'
    rg_location = '../Radiographs/'
    #for zzz in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14']:
    for zzz in ['10']:
        # print('creating PCA:')
        # create_pca_model(int(zzz))
        # print(PCA_MODELS)
        origorig = '../Radiographs/{}.tif'.format(zzz)
        mskfile = '../predicted/predicted_mask{}.tif.tif'.format(zzz)
        #scale_mask_back(mskfile, origorig)
        skip_rg = int(zzz)
        du, dl = scale_mask_back(mskfile, origorig, skip_rg)
        together = du + dl
        print('saving results')
        with open('initial_lmks_{}.cpkl'.format(zzz), 'wb') as fw:
            pickle.dump(together, fw)
