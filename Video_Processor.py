# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:51:40 2020

@author: Dawid
"""

import cv2
assert int(cv2.__version__[0]) >= 3, 'The fisheye module requires opencv version >= 3.0.0'
import numpy
import os
import glob
import time
import sys

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
  
def create_maps_using_homography_matrix(source_image, homography_matrix, invert_matrix = False, destination_image = None):
    """creates map to convert source_image to destination_image using
    destination_image = cv2.remap(source_image, map_x, map_y, cv2.INTER_LINEAR)
    https://stackoverflow.com/questions/46520123/how-do-i-use-opencvs-remap-function"""
    # ground truth homography from destination_image to source_image
    if invert_matrix is True:
        _, homography_matrix = cv2.invert(homography_matrix)
        
    if destination_image is None:
        destination_image = cv2.warpPerspective(source_image, homography_matrix, source_image.shape[:2][::-1])

    # create indices of the destination image and linearize them
    height, width = destination_image.shape[:2]
    indy, indx = numpy.indices((height, width), dtype=numpy.float32)
    linearized_homeography_indices = numpy.array([indx.ravel(), indy.ravel(), numpy.ones_like(indx).ravel()])
    
    # warp the coordinates of source_image to those of destination_image
    map_ind = homography_matrix.dot(linearized_homeography_indices)
    map_x, map_y = map_ind[:-1]/map_ind[-1]  # ensure homogeneity
    map_x = map_x.reshape(height, width).astype(numpy.float32)
    map_y = map_y.reshape(height, width).astype(numpy.float32)

    # convert maps to fixed point numbers
    map_x_fp, map_y_fp = cv2.convertMaps(map_x, map_y, cv2.CV_32FC1)

    return map_x_fp, map_y_fp

def find_parameters_to_make_top_view(img):
    def emptyFunction(newValue):
        pass
    
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 900, 140)
    cv2.createTrackbar("Shrinking parameter", "TrackBars", 0,int(img.shape[1]/2-1),emptyFunction)
    cv2.createTrackbar("Crop top", "TrackBars", 0,img.shape[0],emptyFunction)
    cv2.createTrackbar("Crop bottom", "TrackBars", 0,img.shape[0],emptyFunction)
    
    while True:
        #adjust parameters
        shrinking_parameter = cv2.getTrackbarPos("Shrinking parameter", "TrackBars")
        crop_top = cv2.getTrackbarPos("Crop top", "TrackBars")
        crop_bottom = cv2.getTrackbarPos("Crop bottom", "TrackBars")
        cv2.setTrackbarMax("Crop top", "TrackBars", int(img.shape[0]-crop_bottom-5))
        cv2.setTrackbarMax("Crop bottom", "TrackBars", int(img.shape[0]-crop_top-5))
        
        crop_top = min(int(img.shape[0]-crop_bottom-5), crop_top)
        crop_bottom = min(int(img.shape[0]-crop_top-5), crop_bottom)
        cv2.setTrackbarPos("Crop top", "TrackBars", crop_top)
        cv2.setTrackbarPos("Crop bottom", "TrackBars", crop_bottom)
        
        #calculate from parameters
        heigth_original, width = img.shape[:2]
        img2=img[crop_top:img.shape[0]-crop_bottom,:,:]
        heigth = heigth_original-crop_top-crop_bottom
        points_destination = numpy.array([[0,0], [width,0], [shrinking_parameter,heigth], [width-shrinking_parameter,heigth]], dtype=numpy.float32)
        
        #show images
        top_view_image, map_x, map_y = make_top_view(img2, points_destination=points_destination, return_maps=True)
        cv2.imshow("stackedTwoImages", top_view_image)
        
        keyInput = cv2.waitKey(33)
        if keyInput == 27 or keyInput == 113:    # Esc key to stop
            break
        elif keyInput == -1:  # normally -1 returned,so don't print it
            continue
        # else:
        #     print(keyInput) # else print its value
        
    cv2.destroyWindow("TrackBars")
    cv2.destroyWindow("stackedTwoImages")
    
    return shrinking_parameter, crop_top, crop_bottom, map_x, map_y

def make_top_view(img, points_source = None, points_destination = None, shrinking_parameter = None, crop_top = None, crop_bottom = None, map_x = None, map_y = None, return_maps=False):
    # start_time = time.time()
    if shrinking_parameter is None and crop_top is None and crop_bottom is None:
        # top view from points
        if points_source is None:
            #szukanie jakies sensowne? albo cos z gory zalozyc
            heigth, width = img.shape[:2]
            #bottom left, center left, center right, bottom right
            # points_source = numpy.array([[0, heigth], [int(1/4*width), int(heigth/2)], [int(3/4*width), int(heigth/2)], [width, heigth]], dtype=numpy.float32)
            # top left, top right, bottom right, botom left
            points_source = numpy.array([[0,0], [width,0], [0,heigth], [width,heigth]], dtype=numpy.float32)
            # for pt in points_source:
            #     cv2.circle(img, tuple(pt.astype(numpy.int)), 5, (0,0,255), -1)
        if points_destination is None:
            # points_destination = numpy.array([[0, heigth], [0, 0], [width, 0], [width, heigth]], dtype=numpy.float32) # jakies kompletnie losowe tera
            points_destination = numpy.array([[448,609], [580,609], [580,741], [448,741]], dtype=numpy.float32) # jakies kompletnie losowe tera
            
        if map_x is None or map_y is None:
            homography_matrix = cv2.getPerspectiveTransform(points_source, points_destination)
            map_x, map_y = create_maps_using_homography_matrix(img, homography_matrix, invert_matrix=True)
        result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, cv2.CV_32FC1)
        #result = cv2.warpPerspective(img, homography_matrix, img.shape[:2][::-1])
    else:
        if map_x is None or map_y is None:
            # top view from parameters
            heigth_original, width = img.shape[:2]
            # img2=img[crop_top:img.shape[0]-crop_bottom,:,:]
            heigth = heigth_original-crop_top-crop_bottom
            points_source = numpy.array([[0,0], [width,0], [0,heigth], [width,heigth]], dtype=numpy.float32)
            points_destination = numpy.array([[0,0], [width,0], [shrinking_parameter,heigth], [width-shrinking_parameter,heigth]], dtype=numpy.float32)
        
            homography_matrix = cv2.getPerspectiveTransform(points_source, points_destination)
            map_x, map_y = create_maps_using_homography_matrix(img[crop_top:img.shape[0]-crop_bottom,:,:], homography_matrix, invert_matrix=True)
        result = cv2.remap(img[crop_top:img.shape[0]-crop_bottom,:,:], map_x, map_y, cv2.INTER_LINEAR, cv2.CV_32FC1)
        # result = cv2.warpPerspective(img[crop_top:img.shape[0]-crop_bottom,:,:], homography_matrix, (img.shape[1], heigth))
            
    # display (or save) images
    # cv2.imshow('image', img)
    # cv2.setMouseCallback('image', click_event)
    # cv2.imshow('result', result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # sys.exit()
    # duration1 = time.time()
    # print("Time in make_top_view {:.4f} {:.4f} seconds".format(duration1-start_time, 1.0))
    if return_maps is True:
        return result, map_x, map_y
    else:
        return result

def getCameraParameters(images_paths):
    CHECKERBOARD = (6,8)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    objp = numpy.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), numpy.float32)
    objp[0,:,:2] = numpy.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    i=0
    for fname in images_paths:
        img = cv2.imread(resource_path(fname))
        print("Analyzing " + str(i))
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            print("Found in " + str(i))
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
        i=i+1
    N_OK = len(objpoints)
    K = numpy.zeros((3, 3))
    D = numpy.zeros((4, 1))
    rvecs = [numpy.zeros((1, 1, 3), dtype=numpy.float64) for i in range(N_OK)]
    tvecs = [numpy.zeros((1, 1, 3), dtype=numpy.float64) for i in range(N_OK)]
    rms, K, D, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=numpy.array(" + str(K.tolist()) + ")")
    print("D=numpy.array(" + str(D.tolist()) + ")")
    DIM = _img_shape[::-1]
    return K,D,DIM

def find_undistortion_maps(img, K,D,DIM, balance=0.0, dim2=None, dim3=None):
    # balance not used
    """https://stackoverflow.com/a/44009548"""
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    if not dim3:
        dim3 = dim1
        
    # Scaling the matrix coefficients!
    new_K = K.copy()
    new_K[0,0]=K[0,0]/2
    new_K[1,1]=K[1,1]/2
    
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, numpy.eye(3), new_K, dim3, cv2.CV_16SC2)
    return (map1, map2)

def undistort_with_maps(img, maps):
    """https://stackoverflow.com/a/44009548"""
    undistorted_img = cv2.remap(img, maps[0], maps[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def get_translation2(tx, ty):
    return numpy.float32([
        [1, 0, tx],
        [0, 1, ty],
    ])

def find_parameters_for_combine_top_view(imgBack, imgLeft, imgFront, imgRight, Back_position=None, Left_position=None, Front_position=None, Right_position=None):
    def emptyFunction(newValue):
        pass
    
    #In // whole module Video_Processor and ? // tests "vertical in reality means horizontal" and "horizontal in reality means vertical"

    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 1000, 500)
    
    if Back_position is None or Left_position is None or Front_position is None or Right_position is None:
        cv2.createTrackbar("vertical_offset_for_parallel", "TrackBars", int(imgFront.shape[1]/2),int(imgFront.shape[1]*3/4),emptyFunction)
        cv2.createTrackbar("horizontal_offset_for_parallel", "TrackBars", int(imgLeft.shape[1]/4),int(imgLeft.shape[1]/2),emptyFunction)
        cv2.createTrackbar("vertical_offset_for_perpendicular", "TrackBars", int(imgLeft.shape[0]/2),int(imgLeft.shape[0]),emptyFunction)
        cv2.createTrackbar("horizontal_offset_for_perpendicular", "TrackBars", int(imgFront.shape[0]/2),int(imgFront.shape[0]*3/4),emptyFunction)
        
        cv2.createTrackbar("vertical_scale_for_parallel", "TrackBars", 50,80,emptyFunction)
        cv2.createTrackbar("horizontal_scale_for_parallel", "TrackBars", 50,80,emptyFunction)
        cv2.createTrackbar("vertical_scale_for_perpendicular", "TrackBars", 50,80,emptyFunction)
        cv2.createTrackbar("horizontal_scale_for_perpendicular", "TrackBars", 50,80,emptyFunction)
        
    else:
        cv2.createTrackbar("vertical_offset_for_parallel", "TrackBars", int(Back_position[0]),int(imgFront.shape[1]*3/4),emptyFunction)
        cv2.createTrackbar("horizontal_offset_for_parallel", "TrackBars", int(Back_position[1]),int(imgLeft.shape[1]/2),emptyFunction)
        cv2.createTrackbar("vertical_offset_for_perpendicular", "TrackBars", int(Right_position[0]),int(imgLeft.shape[0]),emptyFunction)
        cv2.createTrackbar("horizontal_offset_for_perpendicular", "TrackBars", int(Right_position[1]),int(imgFront.shape[0]*3/4),emptyFunction)
        
        cv2.createTrackbar("vertical_scale_for_parallel", "TrackBars", int(Back_position[6]),80,emptyFunction)
        cv2.createTrackbar("horizontal_scale_for_parallel", "TrackBars", int(Back_position[7]),80,emptyFunction)
        cv2.createTrackbar("vertical_scale_for_perpendicular", "TrackBars", int(Right_position[6]),80,emptyFunction)
        cv2.createTrackbar("horizontal_scale_for_perpendicular", "TrackBars", int(Right_position[7]),80,emptyFunction)

    # add minimal limits
    cv2.setTrackbarMin("vertical_offset_for_parallel", "TrackBars", int(imgFront.shape[1]/4))
    cv2.setTrackbarMin("horizontal_offset_for_parallel", "TrackBars", int(imgLeft.shape[1]*0.2))
    cv2.setTrackbarMin("vertical_offset_for_perpendicular", "TrackBars", int(imgLeft.shape[0]*0.4))
    cv2.setTrackbarMin("horizontal_offset_for_perpendicular", "TrackBars", int(imgFront.shape[0]/4))
    
    cv2.setTrackbarMin("vertical_scale_for_parallel", "TrackBars", 20)
    cv2.setTrackbarMin("horizontal_scale_for_parallel", "TrackBars", 20)
    cv2.setTrackbarMin("vertical_scale_for_perpendicular", "TrackBars", 20)
    cv2.setTrackbarMin("horizontal_scale_for_perpendicular", "TrackBars", 20)

    while True:
        #adjust parameters
        vertical_offset_for_parallel = cv2.getTrackbarPos("vertical_offset_for_parallel", "TrackBars") - int(imgFront.shape[1]/2)
        horizontal_offset_for_parallel = cv2.getTrackbarPos("horizontal_offset_for_parallel", "TrackBars") - int(imgLeft.shape[1]/4)
        vertical_offset_for_perpendicular = cv2.getTrackbarPos("vertical_offset_for_perpendicular", "TrackBars") - int(imgLeft.shape[0]/2)
        horizontal_offset_for_perpendicular = cv2.getTrackbarPos("horizontal_offset_for_perpendicular", "TrackBars") - int(imgFront.shape[0]/2)
        
        vertical_scale_for_parallel = cv2.getTrackbarPos("vertical_scale_for_parallel", "TrackBars")/50
        horizontal_scale_for_parallel = cv2.getTrackbarPos("horizontal_scale_for_parallel", "TrackBars")/50
        vertical_scale_for_perpendicular = cv2.getTrackbarPos("vertical_scale_for_perpendicular", "TrackBars")/50
        horizontal_scale_for_perpendicular = cv2.getTrackbarPos("horizontal_scale_for_perpendicular", "TrackBars")/50
        
        #calculate from parameters
        Back_position = [vertical_offset_for_parallel,  horizontal_offset_for_parallel,  0,  \
                 0,  0,  0,     \
                     vertical_scale_for_parallel, horizontal_scale_for_parallel]
        Left_position = [-vertical_offset_for_perpendicular,  horizontal_offset_for_perpendicular,  0,  \
                 0,  0,  0,     \
                     vertical_scale_for_perpendicular, horizontal_scale_for_perpendicular]
        Front_position = [vertical_offset_for_parallel,  -horizontal_offset_for_parallel,  0,  \
                 0,  0,  0,     \
                     vertical_scale_for_parallel, horizontal_scale_for_parallel]
        Right_position = [vertical_offset_for_perpendicular,  horizontal_offset_for_perpendicular,  0,  \
                 0,  0,  0,     \
                     vertical_scale_for_perpendicular, horizontal_scale_for_perpendicular]

        #show images
        combined_top_view = combine_top_view(imgBack, imgLeft, imgFront, imgRight, Back_position, Left_position, Front_position, Right_position, return_masks = False, first_run = True)
        cv2.imshow("combined_top_view", cv2.resize(combined_top_view, (0, 0), None, 0.8, 0.8))
        
        keyInput = cv2.waitKey(33)
        if keyInput == 27 or keyInput == 113:    # Esc key to stop
            break
        elif keyInput == -1:  # normally -1 returned,so don't print it
            continue
        # else:
        #     print(keyInput) # else print its value
        
    cv2.destroyWindow("TrackBars")
    cv2.destroyWindow("combined_top_view")
    
    mask_Back, mask_Left, mask_Front, mask_Right, mask_blur = combine_top_view(imgBack, imgLeft, imgFront, imgRight, Back_position, Left_position, Front_position, Right_position, return_masks = True, first_run = False)
    
    return Back_position, Left_position, Front_position, Right_position, mask_Back, mask_Left, mask_Front, mask_Right, mask_blur


def combine_top_view(imgBack, imgLeft, imgFront, imgRight, Back_position=None, Left_position=None, Front_position=None, Right_position=None, mask_Back=None, mask_Left=None, mask_Front=None, mask_Right=None, mask_blur=None, return_masks = False, first_run = False):
    #Position is a list containing x_position, y_position, z_position,
    #                              camera_pitch, camera_yaw, camera_roll,
    #                              image_scale_x, image_scale_y
    # top left is begining x> y\/
    
    # rotation
    imgBack_rotated = cv2.rotate(imgBack, cv2.ROTATE_180)
    imgLeft_rotated = cv2.rotate(imgLeft, cv2.ROTATE_90_COUNTERCLOCKWISE)
    imgRight_rotated = cv2.rotate(imgRight, cv2.ROTATE_90_CLOCKWISE)
    
    if Back_position is None:
        Back_position = [0,  5,  0,  \
                         0,  0,  0,     \
                             1, 0.4]
    if Left_position is None:
        Left_position = [-0,  0,  0,  \
                         0,  0,  0,     \
                             0.4, 1]
    if Front_position is None:
        Front_position = [0,  -5,  0,  \
                         0,  0,  0,     \
                             1, 0.4]
    if Right_position is None:
        Right_position = [0,  0,  0,  \
                         0,  0,  0,     \
                             0.4, 1]
    
    # scale
    imgBack_rotated = cv2.resize(imgBack_rotated, (0, 0), None, Back_position[6], Back_position[7])
    imgLeft_rotated = cv2.resize(imgLeft_rotated, (0, 0), None, Left_position[6], Left_position[7])
    imgFront = cv2.resize(imgFront, (0, 0), None, Front_position[6], Front_position[7])
    imgRight_rotated = cv2.resize(imgRight_rotated, (0, 0), None, Right_position[6], Right_position[7])
    
    calculate_masks = False
    if first_run is True:
        imgCombined_height = max(imgFront.shape[0] - Front_position[1] + imgBack_rotated.shape[0] + Back_position[1], imgLeft_rotated.shape[0], imgRight_rotated.shape[0])
        imgCombined_width = max(imgRight_rotated.shape[1] + Right_position[0] + imgLeft_rotated.shape[1] - Left_position[0], imgFront.shape[1], imgBack_rotated.shape[1])
        center_x = int(imgCombined_width/2)
        center_y = int(imgCombined_height/2)
        
        M_Back = get_translation2(center_x-int(imgBack_rotated.shape[1]/2)+Back_position[0], center_y+Back_position[1])
        M_Left = get_translation2(center_x-int(imgLeft_rotated.shape[1])+Left_position[0], center_y-int(imgLeft_rotated.shape[0]/2)+Left_position[1])
        M_Front = get_translation2(center_x-int(imgFront.shape[1]/2)+Front_position[0], center_y-int(imgFront.shape[0])+Front_position[1])
        M_Right = get_translation2(center_x+Right_position[0], center_y-int(imgRight_rotated.shape[0]/2)+Right_position[1])
        
        
        warpedBack = cv2.warpAffine(imgBack_rotated, M_Back, (imgCombined_width, imgCombined_height))
        warpedLeft = cv2.warpAffine(imgLeft_rotated, M_Left, (imgCombined_width, imgCombined_height))
        warpedFront = cv2.warpAffine(imgFront, M_Front, (imgCombined_width, imgCombined_height))
        warpedRight = cv2.warpAffine(imgRight_rotated, M_Right, (imgCombined_width, imgCombined_height))
        
        calculate_masks = True
        
    else:
        
        # calculate imgCombined size
        imgCombined_height = min(imgFront.shape[0] - Front_position[1] + imgBack_rotated.shape[0] + Back_position[1], imgLeft_rotated.shape[0], imgRight_rotated.shape[0])
        imgCombined_width = min(imgRight_rotated.shape[1] + Right_position[0] + imgLeft_rotated.shape[1] - Left_position[0], imgFront.shape[1], imgBack_rotated.shape[1])
    
        center_y = int(imgCombined_height/2)
        center_x = int(imgCombined_width/2)
        
        # translation
        M_Back = (center_x-int(imgBack_rotated.shape[1]/2)+Back_position[0], center_y+Back_position[1])
        M_Left = (center_x-int(imgLeft_rotated.shape[1])+Left_position[0], center_y-int(imgLeft_rotated.shape[0]/2)+Left_position[1])
        M_Front = (center_x-int(imgFront.shape[1]/2)+Front_position[0], center_y-int(imgFront.shape[0])+Front_position[1],0)
        M_Right = (center_x+Right_position[0], center_y-int(imgRight_rotated.shape[0]/2)+Right_position[1])
        
        warpedBack = numpy.zeros((imgCombined_height, imgCombined_width, 3))
        warpedLeft = numpy.zeros((imgCombined_height, imgCombined_width, 3))
        warpedFront = numpy.zeros((imgCombined_height, imgCombined_width, 3))
        warpedRight = numpy.zeros((imgCombined_height, imgCombined_width, 3))
        
        warpedBack[M_Back[1] : M_Back[1] + imgBack_rotated.shape[0], :, : ] = imgBack_rotated[:, int(imgBack_rotated.shape[1]/2) - Back_position[0] - center_x : int(imgBack_rotated.shape[1]/2) - Back_position[0] + center_x , :]
        warpedFront[M_Front[1] : M_Front[1] + imgFront.shape[0], :, : ] = imgFront[:, int(imgFront.shape[1]/2) - Front_position[0] - center_x : int(imgFront.shape[1]/2) - Front_position[0] + center_x , :]
        warpedLeft[:, M_Left[0] : M_Left[0] + imgLeft_rotated.shape[1], : ] = imgLeft_rotated[int(imgLeft_rotated.shape[0]/2) - Left_position[1] - center_y : int(imgLeft_rotated.shape[0]/2) - Left_position[1] + center_y , :, :]
        warpedRight[:, M_Right[0] : M_Right[0] + imgRight_rotated.shape[1], : ] = imgRight_rotated[int(imgRight_rotated.shape[0]/2) - Right_position[1] - center_y : int(imgRight_rotated.shape[0]/2) - Right_position[1] + center_y , :, :]
        

    if return_masks is True or calculate_masks is True:
        mask_L = (warpedLeft[:,:,0] != 0)
        mask_R = (warpedRight[:,:,0] != 0)
        # type conversion
        mask_L = mask_L.astype(numpy.uint8)
        mask_R = mask_R.astype(numpy.uint8)
        
        # closing
        kernel = numpy.ones((10,10), dtype=numpy.uint8)
        mask_L = cv2.morphologyEx(src=mask_L, op=cv2.MORPH_CLOSE, kernel=kernel)
        mask_R = cv2.morphologyEx(src=mask_R, op=cv2.MORPH_CLOSE, kernel=kernel)
        if first_run is True:
            mask_L = cv2.morphologyEx(src=mask_L, op=cv2.MORPH_CLOSE, kernel=numpy.ones((35,35), dtype=numpy.uint8), borderType=cv2.BORDER_ISOLATED)
            mask_R = cv2.morphologyEx(src=mask_R, op=cv2.MORPH_CLOSE, kernel=numpy.ones((35,35), dtype=numpy.uint8), borderType=cv2.BORDER_ISOLATED)
        else:
            mask_L = cv2.morphologyEx(src=mask_L, op=cv2.MORPH_CLOSE, kernel=numpy.ones((20,20), dtype=numpy.uint8), borderType=cv2.BORDER_ISOLATED)
            mask_R = cv2.morphologyEx(src=mask_R, op=cv2.MORPH_CLOSE, kernel=numpy.ones((20,20), dtype=numpy.uint8), borderType=cv2.BORDER_ISOLATED)
        # erosion
        mask_L = cv2.erode(src=mask_L, kernel=kernel, iterations = 1)
        mask_R = cv2.erode(src=mask_R, kernel=kernel, iterations = 1)
        # prepare side masks
        ones = numpy.ones((int(imgCombined_height/2), imgCombined_width), dtype=numpy.uint8)
        zeros = numpy.zeros((imgCombined_height - int(imgCombined_height/2), imgCombined_width), dtype=numpy.uint8)
        
        mask_B = numpy.vstack((ones,zeros))
        mask_F = numpy.vstack((zeros,ones))
        
        mask_B = numpy.logical_or.reduce((mask_B, mask_L, mask_R))
        mask_F = numpy.logical_or.reduce((mask_F, mask_L, mask_R))
        
        mask_B = numpy.logical_not(mask_B)
        mask_F = numpy.logical_not(mask_F)
        
        # shape conversion
        mask_Back = numpy.stack((mask_B,mask_B,mask_B), axis=2).astype(bool)
        mask_Front = numpy.stack((mask_F,mask_F,mask_F), axis=2).astype(bool)
        mask_Left = numpy.stack((mask_L,mask_L,mask_L), axis=2).astype(bool)
        mask_Right = numpy.stack((mask_R,mask_R,mask_R), axis=2).astype(bool)
        
        # create mask_blur
        mask_B = cv2.dilate(mask_B.astype(numpy.uint8), kernel, iterations = 1)
        mask_F = cv2.dilate(mask_F.astype(numpy.uint8), kernel, iterations = 1)
        mask_L = cv2.dilate(mask_L.astype(numpy.uint8), kernel, iterations = 1)
        mask_R = cv2.dilate(mask_R.astype(numpy.uint8), kernel, iterations = 1)
        mask_blur = numpy.logical_xor.reduce((mask_B, mask_F, mask_L, mask_R))
        mask_blur = numpy.logical_not(mask_blur)
        mask_blur = cv2.morphologyEx(mask_blur.astype(numpy.uint8), cv2.MORPH_CLOSE, kernel)
        #mask_blur = cv2.dilate(mask_blur, kernel, iterations = 2)
        mask_blur = numpy.stack((mask_blur,mask_blur,mask_blur), axis=2)
        # cv2.imshow("mask_Left", mask_Left.astype(numpy.float32))
        # cv2.imshow("mask_Back", mask_Back.astype(numpy.float32))
        # cv2.imshow("mask_blur", mask_blur.astype(numpy.float32))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # sys.exit()
        mask_blur = mask_blur.astype(bool)
        
    
    # combine
    imgCombined = numpy.zeros((imgCombined_height, imgCombined_width, 3), dtype=numpy.uint8)
    imgCombined[mask_Back] = warpedBack[mask_Back]
    imgCombined[mask_Front] = warpedFront[mask_Front]
    imgCombined[mask_Left] = warpedLeft[mask_Left]
    imgCombined[mask_Right] = warpedRight[mask_Right]
    
    imgBlured = cv2.blur(imgCombined,(8,8))
    imgCombined[mask_blur] = imgBlured[mask_blur]
    
    # time4 = time.time()
    # cv2.imshow("imgBack_rotated", imgBack_rotated)
    # imgFront = cv2.circle(imgFront, (x,y), radius=3, color=(0, 0, 255), thickness=-1)
    # cv2.line(imgCombined, (0,y), (imgCombined.shape[1],y), color=(0, 0, 255), thickness=3)
    # cv2.imshow("imgCombined", imgCombined)
    # cv2.imshow("imgCombined", cv2.resize(imgCombined, (0, 0), None, 0.8, 0.8))
    # cv2.imshow("imgLeft_rotated", imgLeft_rotated)
    # cv2.imshow("imgFront", imgFront)
    # cv2.imshow("imgRight_rotated", imgRight_rotated)
    # cv2.imshow("Wrap Affine Back", warpedBack)
    # cv2.imshow("Wrap Affine Left", warpedLeft)
    # cv2.imshow("Wrap Affine Front", warpedFront)
    # cv2.imshow("Wrap Affine Right", warpedRight)
    # cv2.imshow("img combined", imgCombined0)
    # cv2.setMouseCallback('imgCombined', click_event)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # sys.exit()
    # print("Time in rotate {:.4f} scale {:.4f} trans {:.4f} combine {:.4f} total {:.4f} seconds".format(time1-time0, time2-time1, time3-time2, time4-time3, time4-time0))
         
    if return_masks:
        return mask_Back, mask_Left, mask_Front, mask_Right, mask_blur
    else:
        return imgCombined

def equirect_proj(x_proj, y_proj, W, H, fov):
    """Return the equirectangular projection on a unit sphere,
    given cartesian coordinates of the de-warped image."""
    theta_alt = x_proj * fov / W
    phi_alt = y_proj * numpy.pi / H

    x = numpy.sin(theta_alt) * numpy.cos(phi_alt)
    y = numpy.sin(phi_alt)
    z = numpy.cos(theta_alt) * numpy.cos(phi_alt)

    return numpy.arctan2(y, x), numpy.arctan2(numpy.sqrt(x**2 + y**2), z)


def buildmap(Ws, Hs, Wd, Hd, fov=180.0):
    """Return a mapping from de-warped images to fisheye images."""
    fov = fov * numpy.pi / 180.0

    # cartesian coordinates of the de-warped rectangular image
    ys, xs = numpy.indices((Hs, Ws), numpy.float32)
    y_proj = Hs / 2.0 - ys
    x_proj = xs - Ws / 2.0

    # spherical coordinates
    theta, phi = equirect_proj(x_proj, y_proj, Ws, Hs, fov)

    # polar coordinates (of the fisheye image)
    p = Hd * phi / fov

    # cartesian coordinates of the fisheye image
    y_fish = p * numpy.sin(theta)
    x_fish = p * numpy.cos(theta)

    ymap = Hd / 2.0 - y_fish
    xmap = Wd / 2.0 + x_fish
    
    # convert maps to fixed point numbers
    xmap_fp, ymap_fp = cv2.convertMaps(xmap, ymap, cv2.CV_32FC1)
    
    return xmap_fp, ymap_fp


def stack_two_images_with_offsets(img1,img2, offset1, offset2):
    # horizontalSize = img1.shape[0] # lets assume height is the same and dont changes
    # verticalSize = img1.shape[1] - offset1 + img2.shape[1] - offset2
    # depthSize = img1.shape[2]
    # stackedImage = numpy.zeros((horizontalSize, verticalSize), numpy.uint8)
    stackedImage = numpy.concatenate((img1[:,:-offset1,:], img2[:,offset2:,:]), axis=1)
    return stackedImage

def find_parameters_for_two_image_stack(img1, img2, offset1=0, offset2=0):
    def emptyFunction(newValue):
        pass
    
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 900, 80)
    cv2.createTrackbar("Left offset", "TrackBars", offset1, int(img1.shape[1]*0.8), emptyFunction)
    cv2.createTrackbar("Right offset", "TrackBars", offset2, int(img2.shape[1]*0.8), emptyFunction)
    
    cv2.setTrackbarMin("Left offset", "TrackBars", 1)
    cv2.setTrackbarMin("Right offset", "TrackBars", 1)
        
    while True:
        #adjust offset
        offset1 = cv2.getTrackbarPos("Left offset", "TrackBars")
        offset2 = cv2.getTrackbarPos("Right offset", "TrackBars")
        
        #show images
        stackedTwoImages = stack_two_images_with_offsets(img1, img2, offset1, offset2)
        # stackedTwoImages = numpy.hstack((cv2.resize(img1Copy, (0, 0), None, 0.5, 0.5), \
        #                  cv2.resize(img2Copy, (0, 0), None, 0.5, 0.5)))
        cv2.imshow("stackedTwoImages", stackedTwoImages)
        
        
        keyInput = cv2.waitKey(33)
        if keyInput == 27 or keyInput == 113:    # Esc key to stop
            break
        elif keyInput == -1:  # normally -1 returned,so don't print it
            continue
        # else:
        #     print(keyInput) # else print its value
        
    cv2.destroyWindow("TrackBars")
    cv2.destroyWindow("stackedTwoImages")
    
    return offset1, offset2

def warpTwoImages(img1, img2, H = None, parameters = None):
    '''warp img2 to img1 with homograph H'''
    # https://stackoverflow.com/a/20355545
    if H is not None:
        h1,w1 = img1.shape[:2]
        h2,w2 = img2.shape[:2]
        pts1 = numpy.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
        pts2 = numpy.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
        pts2_ = cv2.perspectiveTransform(pts2, H)
        pts = numpy.concatenate((pts1, pts2_), axis=0)
        [xmin, ymin] = numpy.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = numpy.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin,-ymin]
        Ht = numpy.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

        result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
        result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
        parameters = (Ht.dot(H), (xmax-xmin, ymax-ymin), (t[1], h1+t[1], t[0], w1+t[0]))
        return result, parameters
    
    elif parameters is not None:
        result = cv2.warpPerspective(img2, parameters[0], parameters[1])
        result[parameters[2][0]:parameters[2][1],parameters[2][2]:parameters[2][3]] = img1
        return result, parameters
    
    else:
        raise SyntaxError("no H or parameters provided")

class Video_Processor():

    def __init__(self, flags, config_file_path):
        self.USE_PREDEFINED_CAMERA_PARAMETERS = flags[0]
        self.ONLY_VALID_IMAGES_FOR_CAMERA_CALIBRATION = flags[1]
        self.CAMERA_READ_FROM_FILE = flags[2]
        self.USE_PREDEFINED_TOP_VIEW_PARAMETERS = flags[3]
        self.USE_PREDEFINED_COMBINE_TOP_VIEW_PARAMETERS = flags[4]
        self.USE_PREDEFINED_EQUIRECTANGULAR_PARAMETERS = flags[5]
        self.SHOW_IMAGES = flags[6]
        
        config_file_path = resource_path(config_file_path)
        self.load_config_file(config_file_path)

        self.K = None
        self.D = None
        self.DIM = None
        # self.write_images = True
        
        
        self.load_cameras()
        self.frame_counter = 0
        self.calibrate_top_view()
        self.calibrate_equirectangular_projection()
        
    def load_config_file(self, config_file_path):
        self.predefined_camera_parameters_string = [] # USE_PREDEFINED_CAMERA_PARAMETERS
        self.predefined_top_view_parameters_string = [] # USE_PREDEFINED_TOP_VIEW_PARAMETERS
        self.predefined_combine_top_view_parameters_string = [] # USE_PREDEFINED_COMBINE_TOP_VIEW_PARAMETERS
        self.predefined_equirectangular_parameters_string = [] # USE_PREDEFINED_EQUIRECTANGULAR_PARAMETERS
        with open(config_file_path) as file:
            for index, line in enumerate(file):
                # exec(line.strip())
                # print("Line{}: {}".format(index, line.strip()))
                if index <= 2:
                    self.predefined_camera_parameters_string.append(line.strip())
                if index > 2 and index <= 5:
                    self.predefined_top_view_parameters_string.append(line.strip())
                if index > 5 and index <= 9:
                    self.predefined_combine_top_view_parameters_string.append(line.strip())
                if index > 9 and index <= 17:
                    self.predefined_equirectangular_parameters_string.append(line.strip())
                if index == 18:
                    calibration_directory = line.strip()
                    self.calibration_directory = calibration_directory[calibration_directory.rfind('=')+2:]
        print("Read config file")
                    
    def save_config_file(self, config_file_path, new_calibration_directory=None):
        # USE_PREDEFINED_CAMERA_PARAMETERS
        K = "self.K = numpy.array([[" + str(self.K[0][0]) + ", " + str(self.K[0][1]) + ", " + str(self.K[0][2]) + "], [" + \
            str(self.K[1][0]) + ", " + str(self.K[1][1]) + ", " + str(self.K[1][2]) + "], [" + \
            str(self.K[2][0]) + ", " + str(self.K[2][1]) + ", " + str(self.K[2][2]) + "]])\n"
        D = "self.D = numpy.array([[" + str(self.D[0]) + "], [" + str(self.D[1]) + "], [" + str(self.D[2]) + "], [" + str(self.D[3]) + "]])\n"
        
        predefined_camera_parameters_string = (K,
                                               D,
                                               "self.DIM = " + str(self.DIM) + "\n")
        # USE_PREDEFINED_TOP_VIEW_PARAMETERS
        predefined_top_view_parameters_string = ("self.shrinking_parameter = " + str(self.shrinking_parameter) + "\n",
                                                 "self.crop_top = " + str(self.crop_top) + "\n",
                                                 "self.crop_bottom = " + str(self.crop_bottom) + "\n")
        # USE_PREDEFINED_COMBINE_TOP_VIEW_PARAMETERS
        predefined_combine_top_view_parameters_string = ("self.Back_position = " + str(self.Back_position) + "\n",
                                                         "self.Left_position = " + str(self.Left_position) + "\n",
                                                         "self.Front_position = " + str(self.Front_position) + "\n",
                                                         "self.Right_position = " + str(self.Right_position) + "\n")
        # USE_PREDEFINED_EQUIRECTANGULAR_PARAMETERS
        predefined_equirectangular_parameters_string = ("self.offsetBackLeft1 = " + str(self.offsetBackLeft1) + "\n",
                                                        "self.offsetBackLeft2 = " + str(self.offsetBackLeft2) + "\n",
                                                        "self.offsetLeftFront1 = " + str(self.offsetLeftFront1) + "\n",
                                                        "self.offsetLeftFront2 = " + str(self.offsetLeftFront2) + "\n",
                                                        "self.offsetFrontRight1 = " + str(self.offsetFrontRight1) + "\n",
                                                        "self.offsetFrontRight2 = " + str(self.offsetFrontRight2) + "\n",
                                                        "self.offsetRightBack1 = " + str(self.offsetRightBack1) + "\n",
                                                        "self.offsetRightBack2 = " + str(self.offsetRightBack2) + "\n")

        # calibration_directory = self.calibration_directory
        # calibration_directory.replace('\\', '/')
        # directory = calibration_directory[:calibration_directory.rfind("/")]
        if new_calibration_directory is None:
            path = self.calibration_directory
        else:
            path = new_calibration_directory

        with open(config_file_path, "w") as file:
            file.writelines(predefined_camera_parameters_string)
            file.writelines(predefined_top_view_parameters_string)
            file.writelines(predefined_combine_top_view_parameters_string)
            file.writelines(predefined_equirectangular_parameters_string)
            # file.writelines(directory+str(config_file_path))
            file.writelines("calibrationDirectory = " + str(path))
        print("Saved new config file")
        
        
    def load_cameras(self):
        if self.CAMERA_READ_FROM_FILE is True:
            # DO_ZMIANY
            self.capBack = cv2.VideoCapture(resource_path('260-290mp4\\Back_0260-0290.mp4'))
            self.capLeft = cv2.VideoCapture(resource_path('260-290mp4\\Left_0260-0290.mp4'))
            self.capFront = cv2.VideoCapture(resource_path('260-290mp4\\Front_0260-0290.mp4'))
            self.capRight = cv2.VideoCapture(resource_path('260-290mp4\\Right_0260-0290.mp4'))
            # self.capBack = cv2.VideoCapture(resource_path('100-110mp4\\Back_0100-0110.mp4'))
            # self.capLeft = cv2.VideoCapture(resource_path('100-110mp4\\Left_0100-0110.mp4'))
            # self.capFront = cv2.VideoCapture(resource_path('100-110mp4\\Front_0100-0110.mp4'))
            # self.capRight = cv2.VideoCapture(resource_path('100-110mp4\\Right_0100-0110.mp4'))
        else:
            self.capBack = cv2.VideoCapture(1)
            self.capLeft = cv2.VideoCapture(2)
            self.capFront = cv2.VideoCapture(3)
            self.capRight = cv2.VideoCapture(4)
        
        if cv2.VideoCapture.isOpened(self.capBack) is False or  \
            cv2.VideoCapture.isOpened(self.capLeft) is False or \
            cv2.VideoCapture.isOpened(self.capFront) is False or \
            cv2.VideoCapture.isOpened(self.capRight) is False:
        
                raise SystemExit("Camera not found")
                
        
       
    def calibrate_camera(self):
        """
        Calibrate camera
        """
        if self.USE_PREDEFINED_CAMERA_PARAMETERS is True:
            # self.K = numpy.array([[219.85077387813544, 0.0, 321.8468539428703], [0.0, 219.81115217715458, 321.26199300586325], [0.0, 0.0, 1.0]])
            # self.D = numpy.array([[-0.02236163741176025], [-0.01566355538478192], [0.0066695817100666304], [-0.0009867103996664935]])
            # self.K=numpy.array([[219.5428010096495, 0.0, 321.15662214737426], [0.0, 219.4320427696042, 321.38504973935045], [0.0, 0.0, 1.0]])
            # self.D=numpy.array([[-0.020000607973027145], [-0.017743359032950514], [0.00814148877228781], [-0.0014091359313970462]])
            # self.DIM =(640, 640)
            for line in self.predefined_camera_parameters_string:
                exec(line)
        elif self.ONLY_VALID_IMAGES_FOR_CAMERA_CALIBRATION is True:
            # DO_ZMIANY
            self.calibration_directory = only_valid_images_for_calibration
            self.K, self.D, self.DIM = getCameraParameters(self.calibration_directory)
        else:
            # self.calibration_directory = 'dataset5\\*.jpg'
            calibration_directory = glob.glob(self.calibration_directory)
            self.K, self.D, self.DIM = getCameraParameters(calibration_directory)
        
        
    def calibrate_top_view(self):
        """
        Calibrate top view
        """
        if self.frame_counter == 0:
            self.read_frame()
        if self.K is None or self.D is None or self.DIM is None:
            self.calibrate_camera()


        #unwrap images using fisheye calibration
        self.undistortion_maps = find_undistortion_maps(self.imgBack,self.K,self.D,self.DIM)
        imgBack_unwarped0 = undistort_with_maps(self.imgBack, self.undistortion_maps)
        imgLeft_unwarped0 = undistort_with_maps(self.imgLeft, self.undistortion_maps)
        imgFront_unwarped0 = undistort_with_maps(self.imgFront, self.undistortion_maps)
        imgRight_unwarped0 = undistort_with_maps(self.imgRight, self.undistortion_maps)
        # img3 = numpy.concatenate((imgBack_unwarped0, imgLeft_unwarped0, imgFront_unwarped0, imgRight_unwarped0), axis=1)

        if self.USE_PREDEFINED_TOP_VIEW_PARAMETERS is True:
            # # 1 version
            # self.shrinking_parameter = 300
            # self.crop_top = 340
            # self.crop_bottom = 0
            for line in self.predefined_top_view_parameters_string:
                exec(line)

            _, self.top_view_map_x, self.top_view_map_y = make_top_view(imgBack_unwarped0, shrinking_parameter=self.shrinking_parameter, crop_top=self.crop_top, crop_bottom=self.crop_bottom, return_maps=True)
        else:
            self.shrinking_parameter, self.crop_top, self.crop_bottom, self.top_view_map_x, self.top_view_map_y = find_parameters_to_make_top_view(imgBack_unwarped0)
 
        imgBack_topview = make_top_view(imgBack_unwarped0, shrinking_parameter=self.shrinking_parameter, crop_top=self.crop_top, crop_bottom=self.crop_bottom, map_x=self.top_view_map_x, map_y=self.top_view_map_y)
        imgLeft_topview = make_top_view(imgLeft_unwarped0, shrinking_parameter=self.shrinking_parameter, crop_top=self.crop_top, crop_bottom=self.crop_bottom, map_x=self.top_view_map_x, map_y=self.top_view_map_y)
        imgFront_topview = make_top_view(imgFront_unwarped0, shrinking_parameter=self.shrinking_parameter, crop_top=self.crop_top, crop_bottom=self.crop_bottom, map_x=self.top_view_map_x, map_y=self.top_view_map_y)
        imgRight_topview = make_top_view(imgRight_unwarped0, shrinking_parameter=self.shrinking_parameter, crop_top=self.crop_top, crop_bottom=self.crop_bottom, map_x=self.top_view_map_x, map_y=self.top_view_map_y)
        # img4 = numpy.concatenate((imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview), axis=1)
        # if self.SHOW_IMAGES is True:
        #     cv2.imshow("Unwarped by undistort in line", cv2.resize(img3, (0, 0), None, 0.5, 0.5))
        #     cv2.imshow("Top view", cv2.resize(img4, (0, 0), None, 0.5, 0.5))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #     sys.exit()
        

        if self.USE_PREDEFINED_COMBINE_TOP_VIEW_PARAMETERS is True:
            # # 1 version
            # self.Back_position = [0, 8, 0, 0, 0, 0, 1.38, 0.56]
            # self.Left_position = [0, 0, 0, 0, 0, 0, 0.54, 1.56]
            # self.Front_position = [0, -8, 0, 0, 0, 0, 1.38, 0.56]
            # self.Right_position = [0, 0, 0, 0, 0, 0, 0.54, 1.56]
            for line in self.predefined_combine_top_view_parameters_string:
                exec(line)
        
            self.mask_Back, self.mask_Left, self.mask_Front, self.mask_Right, self.mask_blur = combine_top_view(imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview, self.Back_position, self.Left_position, self.Front_position, self.Right_position, return_masks=True, first_run=False)
        else:
            vertical_offset_for_parallel = 320
            horizontal_offset_for_parallel = 160
            vertical_offset_for_perpendicular = 150
            horizontal_offset_for_perpendicular = 145
            
            vertical_scale_for_parallel = 76
            horizontal_scale_for_parallel = 31
            vertical_scale_for_perpendicular = 30
            horizontal_scale_for_perpendicular = 77
            
            self.Back_position = [vertical_offset_for_parallel,  horizontal_offset_for_parallel,  0,  \
                      0,  0,  0,     \
                          vertical_scale_for_parallel, horizontal_scale_for_parallel]
            self.Left_position = [-vertical_offset_for_perpendicular,  horizontal_offset_for_perpendicular,  0,  \
                      0,  0,  0,     \
                          vertical_scale_for_perpendicular, horizontal_scale_for_perpendicular]
            self.Front_position = [vertical_offset_for_parallel,  -horizontal_offset_for_parallel,  0,  \
                      0,  0,  0,     \
                          vertical_scale_for_parallel, horizontal_scale_for_parallel]
            self.Right_position = [vertical_offset_for_perpendicular,  horizontal_offset_for_perpendicular,  0,  \
                      0,  0,  0,     \
                          vertical_scale_for_perpendicular, horizontal_scale_for_perpendicular]
    
            self.Back_position, self.Left_position, self.Front_position, self.Right_position, self.mask_Back, self.mask_Left, self.mask_Front, self.mask_Right, self.mask_blur = find_parameters_for_combine_top_view(imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview, self.Back_position, self.Left_position, self.Front_position, self.Right_position)

        
    def calibrate_equirectangular_projection(self):
        """
        Calibrate equirectangular
        """
        if self.frame_counter == 0:
            self.read_frame()
            
            
        # unwarp images using projection
        # W_remap and H are resolution of image converted to equirectangular
        # self.W_remap = 720
        self.W_remap = 640
        self.H = 640
        self.FOV = 180
        self.equirectangular_xmap, self.equirectangular_ymap = buildmap(Ws=self.W_remap, Hs=self.H, Wd=640, Hd=640, fov=self.FOV)
        imgBack_unwarped = cv2.remap(self.imgBack, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)
        imgLeft_unwarped = cv2.remap(self.imgLeft, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)
        imgFront_unwarped = cv2.remap(self.imgFront, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)
        imgRight_unwarped = cv2.remap(self.imgRight, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)
    
        # img4 = numpy.concatenate((imgBack_unwarped, imgLeft_unwarped, imgFront_unwarped, imgRight_unwarped), axis=1)
        
        # if self.SHOW_IMAGES is True:
        #     cv2.imshow("Top View image", combined_top_view)cv2.imshow("Unwarped in line", cv2.resize(img4, (0, 0), None, 0.5, 0.5))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #     sys.exit()
        
                
        
        # Use starting parameters for equirectangular
        # when self.W_remap == 720
        # self.offsetBackLeft1 = 167
        # self.offsetBackLeft2 = 167
        # self.offsetLeftFront1 = 167
        # self.offsetLeftFront2 = 167
        # self.offsetFrontRight1 = 167
        # self.offsetFrontRight2 = 167
        # self.offsetRightBack1 = 167
        # self.offsetRightBack2 = 167
        # # when self.W_remap == 640
        # self.offsetBackLeft1 = 148
        # self.offsetBackLeft2 = 148
        # self.offsetLeftFront1 = 148
        # self.offsetLeftFront2 = 148
        # self.offsetFrontRight1 = 148
        # self.offsetFrontRight2 = 148
        # self.offsetRightBack1 = 148
        # self.offsetRightBack2 = 148
        for line in self.predefined_equirectangular_parameters_string:
            exec(line)
            
        if self.USE_PREDEFINED_EQUIRECTANGULAR_PARAMETERS is False:
            # find parameters
            self.offsetBackLeft1, self.offsetBackLeft2 = find_parameters_for_two_image_stack(imgBack_unwarped[:,int(self.W_remap/2):,:], imgLeft_unwarped[:,:int(self.W_remap/2),:], self.offsetBackLeft1, self.offsetBackLeft2)
            self.offsetLeftFront1, self.offsetLeftFront2 = find_parameters_for_two_image_stack(imgLeft_unwarped[:,int(self.W_remap/2):,:], imgFront_unwarped[:,:int(self.W_remap/2),:], self.offsetLeftFront1, self.offsetLeftFront2)
            self.offsetFrontRight1, self.offsetFrontRight2 = find_parameters_for_two_image_stack(imgFront_unwarped[:,int(self.W_remap/2):,:], imgRight_unwarped[:,:int(self.W_remap/2),:], self.offsetFrontRight1, self.offsetFrontRight2)
            self.offsetRightBack1, self.offsetRightBack2 = find_parameters_for_two_image_stack(imgRight_unwarped[:,int(self.W_remap/2):,:], imgBack_unwarped[:,:int(self.W_remap/2),:], self.offsetRightBack1, self.offsetRightBack2)
         
        # create blur mask
        size_1 = imgRight_unwarped.shape[1]-self.offsetRightBack1-int(self.W_remap/2)
        size_2 = imgBack_unwarped.shape[1]-self.offsetBackLeft1-self.offsetRightBack2
        size_3 = imgLeft_unwarped.shape[1]-self.offsetLeftFront1-self.offsetBackLeft2
        size_4 = imgFront_unwarped.shape[1]-self.offsetFrontRight1-self.offsetLeftFront2
        
        size_5 = int(self.W_remap/2)-self.offsetFrontRight2

        crop_black = int(self.imgBack.shape[0]*0.007)
        seam_width = 10  
        
        mask_blur_equirectangular = numpy.zeros((imgRight_unwarped.shape[0]-2*crop_black, size_1+size_2+size_3+size_4+size_5, 3))
        mask_blur_equirectangular[:,size_1-seam_width:size_1+seam_width,:] = 1
        mask_blur_equirectangular[:,size_1+size_2-seam_width:size_1+size_2+seam_width,:] = 1
        mask_blur_equirectangular[:,size_1+size_2+size_3-seam_width:size_1+size_2+size_3+seam_width,:] = 1
        mask_blur_equirectangular[:,size_1+size_2+size_3+size_4-seam_width:size_1+size_2+size_3+size_4+seam_width,:] = 1
        mask_blur_equirectangular = mask_blur_equirectangular.astype(bool)
        self.mask_blur_equirectangular = mask_blur_equirectangular
            
            
        
    def read_frame(self):
        successBack, self.imgBack = self.capBack.read()
        successLeft, self.imgLeft = self.capLeft.read()
        successFront, self.imgFront = self.capFront.read()
        successRight, self.imgRight = self.capRight.read()
        
        if successBack is True and successLeft is True and successFront is True and successRight is True:
            self.frame_read_successfully = True
            self.frame_counter += 1
        else:
            raise SystemExit("Reading frame failed")
            self.frame_read_successfully = False
            
        if self.CAMERA_READ_FROM_FILE is True:
            #loop video
            if self.frame_counter == self.capBack.get(cv2.CAP_PROP_FRAME_COUNT) or \
                self.frame_counter == self.capLeft.get(cv2.CAP_PROP_FRAME_COUNT) or \
                    self.frame_counter == self.capFront.get(cv2.CAP_PROP_FRAME_COUNT) or \
                        self.frame_counter == self.capRight.get(cv2.CAP_PROP_FRAME_COUNT):
                self.frame_counter = 0
                self.capBack.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.capLeft.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.capFront.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.capRight.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # #make it visible for human eye
            # time.sleep(0.2)
        
    def top_view(self):
        # time0 = time.time()
        #unwrap images using fisheye calibration
        imgBack_unwarped0 = undistort_with_maps(self.imgBack, self.undistortion_maps)
        imgLeft_unwarped0 = undistort_with_maps(self.imgLeft, self.undistortion_maps)
        imgFront_unwarped0 = undistort_with_maps(self.imgFront, self.undistortion_maps)
        imgRight_unwarped0 = undistort_with_maps(self.imgRight, self.undistortion_maps)
        
        # img3 = numpy.concatenate((imgBack_unwarped0, imgLeft_unwarped0, imgFront_unwarped0, imgRight_unwarped0), axis=1)
        # time1 = time.time()
        imgBack_topview = make_top_view(imgBack_unwarped0, shrinking_parameter=self.shrinking_parameter, crop_top=self.crop_top, crop_bottom=self.crop_bottom, map_x=self.top_view_map_x, map_y=self.top_view_map_y)
        imgLeft_topview = make_top_view(imgLeft_unwarped0, shrinking_parameter=self.shrinking_parameter, crop_top=self.crop_top, crop_bottom=self.crop_bottom, map_x=self.top_view_map_x, map_y=self.top_view_map_y)
        imgFront_topview = make_top_view(imgFront_unwarped0, shrinking_parameter=self.shrinking_parameter, crop_top=self.crop_top, crop_bottom=self.crop_bottom, map_x=self.top_view_map_x, map_y=self.top_view_map_y)
        imgRight_topview = make_top_view(imgRight_unwarped0, shrinking_parameter=self.shrinking_parameter, crop_top=self.crop_top, crop_bottom=self.crop_bottom, map_x=self.top_view_map_x, map_y=self.top_view_map_y)

        # time2 = time.time()
        self.top_view_image = combine_top_view(imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview, self.Back_position, self.Left_position, self.Front_position, self.Right_position, self.mask_Back, self.mask_Left, self.mask_Front, self.mask_Right, self.mask_blur)
        # time3 = time.time()
        
        # print("Time undistort {:.4f} top_view {:.4f} combine {:.4f} total {:.4f} seconds".format(time1-time0, time2-time1, time3-time2, time3-time0))
                
        
        # cv2.imwrite("prezentacja/imgBack_unwarped0.jpg", imgBack_unwarped0)
        # cv2.imwrite("prezentacja/imgBack_topview.jpg", imgBack_topview)
        # cv2.imwrite("prezentacja/top_view_image.jpg", self.top_view_image)
        # sys.exit()
        
        # if self.SHOW_IMAGES is True:
        #     cv2.imshow("self.top_view_image", self.top_view_image)
        # #     # cv2.imshow("combined_top_view", combined_top_view)
        #     cv2.waitKey(0)
        #     # cv2.destroyWindow("self.top_view_image")
        #     cv2.destroyAllWindows()
        #     sys.exit()
        return self.top_view_image
        
 
    def equirectangular_projection(self):
        # crop_black should be dependand on camera FOV and resolution
        crop_black = int(self.imgBack.shape[0]*0.007)
        imgBack_unwarped = cv2.remap(self.imgBack, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)[crop_black:-crop_black,:,:]
        imgLeft_unwarped = cv2.remap(self.imgLeft, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)[crop_black:-crop_black,:,:]
        imgFront_unwarped = cv2.remap(self.imgFront, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)[crop_black:-crop_black,:,:]
        imgRight_unwarped = cv2.remap(self.imgRight, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)[crop_black:-crop_black,:,:]
        
        imgConcatenated = numpy.concatenate((
            imgRight_unwarped[:,int(self.W_remap/2):-self.offsetRightBack1,:],
            imgBack_unwarped[:,self.offsetRightBack2:-self.offsetBackLeft1,:],
            imgLeft_unwarped[:,self.offsetBackLeft2:-self.offsetLeftFront1,:],
            imgFront_unwarped[:,self.offsetLeftFront2:-self.offsetFrontRight1,:],
            imgRight_unwarped[:,self.offsetFrontRight2:int(self.W_remap/2),:]
            ), axis=1)

        
        imgBlured = cv2.blur(imgConcatenated,(3,3))
        imgConcatenated[self.mask_blur_equirectangular] = imgBlured[self.mask_blur_equirectangular]
        self.equirectangular_image = imgConcatenated
        # cv2.imshow("imgConcatenated", imgConcatenated.astype(numpy.uint8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # sys.exit()
        # cv2.imwrite("prezentacja/equirectangular_image_square.jpg", self.equirectangular_image)
        # sys.exit()
            
          
        
        # if self.SHOW_IMAGES is True:
        # if self.write_images is True:
        #     cv2.imshow("self.equirectangular_image", self.equirectangular_image)
            # cv2.imwrite("prezentacja/imgBack_unwarped.jpg", imgBack_unwarped)
        #     cv2.imwrite("prezentacja/imgBack.jpg", self.imgBack)
            # cv2.imwrite("prezentacja/equirectangular_image.jpg", self.equirectangular_image)
            # self.write_images = False
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #     sys.exit()
        return self.equirectangular_image
            
            
    def run(self, dont_stop = True):
        if self.frame_counter == 0:
            self.read_frame()
        self.FPS = []
        while dont_stop:
            if cv2.waitKey(1) & 0xFF == ord('q') or self.frame_read_successfully is False or dont_stop is False:
                break
            else:
                time0 = time.time()
                
                self.top_view()
                # time1 = time.time()

                self.equirectangular_projection()
                # time2 = time.time()
                
                self.read_frame()
                # time3 = time.time()
                
                if self.SHOW_IMAGES is True:
                    cv2.imshow("self.top_view_image", self.top_view_image)
                    cv2.imshow("self.equirectangular_image", self.equirectangular_image)
                        
                # if self.write_images is True and self.frame_counter == 27:
                #     cv2.imwrite("prezentacja/top_view_image-5.jpg", self.top_view_image)
                #     cv2.imwrite("prezentacja/equirectangular_image_square-5.jpg", self.equirectangular_image)
                #     self.write_images = False
                        
                    # time.sleep(0.2)

                # Stop run if it was wrongly initialized
                else:
                    if self.frame_counter == 10:
                        dont_stop = False
                                            
                time4 = time.time()
                    
                # print("Time in top_view {:.4f} equi {:.4f} read {:.4f} show {:.4f} total {:.4f} seconds".format(time1-time0, time2-time1, time3-time2, time4-time3, time4-time0))
                self.FPS.append(time4-time0)
                
    
    def __del__(self):
        cv2.destroyAllWindows()
        self.capBack.release()
        self.capLeft.release()
        self.capFront.release()
        self.capRight.release()
        if hasattr(self, 'FPS'):
            print("Average FPS {}".format(1/(sum(self.FPS)/len(self.FPS))))


"""
Start here
"""

if __name__ == '__main__':
    # Declare parameters of program
    USE_PREDEFINED_CAMERA_PARAMETERS = True
    ONLY_VALID_IMAGES_FOR_CAMERA_CALIBRATION = True
    CAMERA_READ_FROM_FILE = True
    
    USE_PREDEFINED_TOP_VIEW_PARAMETERS = True
    USE_PREDEFINED_COMBINE_TOP_VIEW_PARAMETERS = True  # False - first guess to make it easier
    
    USE_PREDEFINED_EQUIRECTANGULAR_PARAMETERS = True # False - first guess to make it easier
    
    SHOW_IMAGES = True
    
    
    if ONLY_VALID_IMAGES_FOR_CAMERA_CALIBRATION is True:
        only_valid_images_for_calibration = []
        only_valid_images_for_calibration.append("dataset5\\0009.jpg")
        only_valid_images_for_calibration.append("dataset5\\0010.jpg")
        only_valid_images_for_calibration.append("dataset5\\0023.jpg")
        only_valid_images_for_calibration.append("dataset5\\0032.jpg")
        only_valid_images_for_calibration.append("dataset5\\0037.jpg")
        only_valid_images_for_calibration.append("dataset5\\0046.jpg")
        only_valid_images_for_calibration.append("dataset5\\0058.jpg")
        only_valid_images_for_calibration.append("dataset5\\0067.jpg")
        only_valid_images_for_calibration.append("dataset5\\0072.jpg")
        only_valid_images_for_calibration.append("dataset5\\0081.jpg")
        only_valid_images_for_calibration.append("dataset5\\0094.jpg")
        
    flags = (USE_PREDEFINED_CAMERA_PARAMETERS,
             ONLY_VALID_IMAGES_FOR_CAMERA_CALIBRATION,
             CAMERA_READ_FROM_FILE,
             USE_PREDEFINED_TOP_VIEW_PARAMETERS,
             USE_PREDEFINED_COMBINE_TOP_VIEW_PARAMETERS,
             USE_PREDEFINED_EQUIRECTANGULAR_PARAMETERS,
             SHOW_IMAGES)
    
    config_file_path = "config.txt"
    
    Video_Processor = Video_Processor(flags, config_file_path)
    # Video_Processor.top_view()
    # Video_Processor.equirectangular_projection()
    Video_Processor.run(dont_stop = True)
    Video_Processor.save_config_file("new_config.txt")
    del Video_Processor
    # try:
    #     Video_Processor()
    # except Exception:
    #     print("type: \t\t", sys.exc_info()[0].__name__, 
    #           "\nfilename: \t", os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1], 
    #           "\nlineo: \t\t", sys.exc_info()[2].tb_lineno,
    #           #"\nname: \t", sys.exc_info()[2].tb_frame.f_code.co_name,
    #           "\nmessage: \t", sys.exc_info()[1])
    