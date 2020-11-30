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
from matplotlib import pyplot as plt
import sys

 
def click_event(event, x, y, flags, params):
    """function to display the coordinates of the points clicked on the image"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)    
    if event==cv2.EVENT_RBUTTONDOWN:
        print(x, ' ', y)

def find_parameters_to_make_top_view(img):
    def emptyFunction(newValue):
        pass
    
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 900, 140)
    cv2.createTrackbar("Shrinking parameter", "TrackBars", 0,int(img.shape[1]/2),emptyFunction)
    cv2.createTrackbar("Crop top", "TrackBars", 0,img.shape[0],emptyFunction)
    cv2.createTrackbar("Crop bottom", "TrackBars", 0,img.shape[0],emptyFunction)
    
    while True:
        #adjust parameters
        shrinking_parameter = cv2.getTrackbarPos("Shrinking parameter", "TrackBars")
        crop_top = cv2.getTrackbarPos("Crop top", "TrackBars")
        crop_bottom = cv2.getTrackbarPos("Crop bottom", "TrackBars")
        
        #calculate from parameters
        heigth_original, width = img.shape[:2]
        img2=img[crop_top:img.shape[0]-crop_bottom,:,:]
        heigth = heigth_original-crop_top-crop_bottom
        points_destination = numpy.array([[0,0], [width,0], [shrinking_parameter,heigth], [width-shrinking_parameter,heigth]], dtype=numpy.float32)
        
        #show images
        top_view_image = make_top_view(img2, points_destination=points_destination)
        cv2.imshow("stackedTwoImages", top_view_image)
        
        keyInput = cv2.waitKey(33)
        if keyInput==27:    # Esc key to stop
            break
        elif keyInput==-1:  # normally -1 returned,so don't print it
            continue
        else:
            print(keyInput) # else print its value
        
    cv2.destroyWindow("TrackBars")
    cv2.destroyWindow("stackedTwoImages")
    
    return shrinking_parameter, crop_top, crop_bottom

def make_top_view(img, points_source = None, points_destination = None, shrinking_parameter = None, crop_top = None, crop_bottom = None):
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
            
        homography_matrix = cv2.getPerspectiveTransform(points_source, points_destination)
        result = cv2.warpPerspective(img, homography_matrix, img.shape[:2][::-1])
    else:
        # top view from parameters
        heigth_original, width = img.shape[:2]
        img2=img[crop_top:img.shape[0]-crop_bottom,:,:]
        heigth = heigth_original-crop_top-crop_bottom
        points_source = numpy.array([[0,0], [width,0], [0,heigth], [width,heigth]], dtype=numpy.float32)
        points_destination = numpy.array([[0,0], [width,0], [shrinking_parameter,heigth], [width-shrinking_parameter,heigth]], dtype=numpy.float32)
        
        homography_matrix = cv2.getPerspectiveTransform(points_source, points_destination)
        result = cv2.warpPerspective(img2, homography_matrix, img2.shape[:2][::-1])
            
    # display (or save) images
    # cv2.imshow('image', img)
    # cv2.setMouseCallback('image', click_event)
    # cv2.imshow('result', result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    # sys.exit()
    return result

def getCameraParameters_using_omnidirectional(images_paths):
    '''https://docs.opencv.org/3.4/dd/d12/tutorial_omnidir_calib_main.html
    https://stackoverflow.com/questions/56942485/python-omnidirectional-calibration
    https://gist.github.com/mesutpiskin/0412c44bae399adf1f48007f22bdd22d'''
    
    """Inny typ kamery :D"""
    CHECKERBOARD = (6,8)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    #calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    calibration_flags = cv2.omnidir.CALIB_FIX_SKEW+cv2.omnidir.CALIB_FIX_CENTER
    objp = numpy.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), numpy.float32)
    objp[0,:,:2] = numpy.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    i=0
    for fname in images_paths:
        img = cv2.imread(fname)
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
            print("Founded in " + str(i))
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
        i=i+1
    N_OK = len(objpoints)
    K = numpy.zeros((3, 3))
    D = numpy.zeros([1, 4])
    xi = numpy.zeros(1)
    rvecs = [numpy.zeros((1, 1, 3), dtype=numpy.float32) for i in range(N_OK)]
    tvecs = [numpy.zeros((1, 1, 3), dtype=numpy.float32) for i in range(N_OK)]
    rms, K, xi, D, rvecs, tvecs, idx = \
        cv2.omnidir.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            #images_shape,
            K,
            xi,
            D,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
            rvecs,
            tvecs
        )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=numpy.array(" + str(K.tolist()) + ")")
    print("D=numpy.array(" + str(D.tolist()) + ")")
     
    print("cv::omnidir::undistortImage")
    distorted = cv2.imread("dataset5/0023.jpg")
    undistorted = distorted
    # Knew = numpy.zeros((3, 3))
    Knew = numpy.array([[_img_shape[1]/4, 0, _img_shape[1]/2],
                        [0, _img_shape[0]/4, _img_shape[1]/2],
                        [0, 0, 1]])
    undistort_flags = cv2.omnidir.RECTIFY_STEREOGRAPHIC
    undistorted = cv2.omnidir.undistortImage(distorted, K, D, xi, undistort_flags, Knew)
    
    new_K = K.copy()
    new_K[0,0]=K[0,0]/2
    new_K[1,1]=K[1,1]/2
    dim3=_img_shape[::-1]
    map1, map2 = cv2.omnidir.initUndistortRectifyMap(K, D, xi, numpy.eye(3), Knew, dim3, cv2.CV_16SC2, int(undistort_flags))  # Pass k in 1st parameter, nk in 4th parameter
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
   
    
    cv2.imshow("distorted elo", distorted)
    cv2.imshow("undistorted elo", undistorted)
    cv2.imshow("undistorted_img", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    DIM=_img_shape[::-1]
    return K,D,DIM

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
        img = cv2.imread(fname)
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
            print("Founded in " + str(i))
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
        i=i+1
    N_OK = len(objpoints)
    K = numpy.zeros((3, 3))
    D = numpy.zeros((4, 1))
    rvecs = [numpy.zeros((1, 1, 3), dtype=numpy.float64) for i in range(N_OK)]
    tvecs = [numpy.zeros((1, 1, 3), dtype=numpy.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
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

# step 2
#DIM=_img_shape[::-1]
def undistort(img, K,D,DIM):
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, numpy.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # cv2.imshow("undistorted", undistorted_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return undistorted_img
    
def undistort2(img, K,D,DIM, balance=0.0, dim2=None, dim3=None):
    # FOV 180 doesnt work good here
    # without croping its not good, results same as 3 in corped
    # balance does not work?
    # dim 3 crops image as said in medium article, but crops from top left, not center, what is dumb
    """https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f"""
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, numpy.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, numpy.eye(3), new_K, dim3, cv2.CV_32FC1)#cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def undistort3(img, K,D,DIM, balance=0.0, dim2=None, dim3=None):
    # so far best
    # balance not used
    """https://stackoverflow.com/a/44009548"""
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    if not dim3:
        dim3 = dim1
        
    # Scaling the matrix coefficients!
    new_K = K.copy()
    new_K[0,0]=K[0,0]/2
    new_K[1,1]=K[1,1]/2
    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, numpy.eye(3), new_K, (800,600), cv2.CV_16SC2)  # Pass k in 1st parameter, nk in 4th parameter
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, numpy.eye(3), new_K, dim3, cv2.CV_16SC2)  # Pass k in 1st parameter, nk in 4th parameter
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def undistort4(img, K,D,DIM, balance=0.0, dim2=None, dim3=None):
    if not dim2:
        dim2 = DIM
    if not dim3:
        dim3 = DIM
    scaled_K = K * dim2[0] / DIM[0]  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim2, numpy.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, numpy.eye(3), new_K, dim3, cv2.CV_32FC1)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def get_rotation_x_clockwise(angle): #chyba clockwise :)
    angle = numpy.radians(angle)
    return numpy.array([
        [1,  0, 0, 0],
        [0, numpy.cos(angle), -numpy.sin(angle), 0],
        [0, numpy.sin(angle),  numpy.cos(angle), 0],
        [0, 0, 0, 1]
    ])
def get_rotation_y_clockwise(angle): #chyba clockwise :)
    angle = numpy.radians(angle)
    return numpy.array([
        [numpy.cos(angle), 0,  numpy.sin(angle), 0],
        [0,  1, 0, 0],
        [-numpy.sin(angle), 0, numpy.cos(angle), 0],
        [0, 0, 0, 1]
    ])
def get_rotation_z_clockwise(angle): #chyba clockwise :)
    angle = numpy.radians(angle)
    return numpy.array([
        [numpy.cos(angle), -numpy.sin(angle), 0, 0],
        [numpy.sin(angle),  numpy.cos(angle), 0, 0],
        [0,  0, 1, 0],
        [0, 0, 0, 1]
    ])
def get_translation(tx, ty, tz):
    return numpy.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
def get_scale(sx, sy, sz):
    return numpy.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])


def get_translation2(tx, ty):
    return numpy.float32([
        [1, 0, tx],
        [0, 1, ty],
    ])

def get_affine_cv(translationXY, rotation, scale):
    """translationXY is touple (x,y), rotation is angle counter clockwise, no scale is 1
    https://towardsdatascience.com/image-geometric-transformation-in-numpy-and-opencv-936f5cd1d315"""
    sin_theta = numpy.sin(rotation)
    cos_theta = numpy.cos(rotation)
    
    a_11 = scale * cos_theta
    a_21 = -scale * sin_theta
    
    a_12 = scale * sin_theta
    a_22 = scale * cos_theta
        
    a_13 = translationXY[0] * (1 - scale * cos_theta) - scale * sin_theta * translationXY[1]
    a_23 = translationXY[1] * (1 - scale * cos_theta) + scale * sin_theta * translationXY[0]
    return numpy.array([[a_11, a_12, a_13],
                     [a_21, a_22, a_23]])
    # A2 = get_affine_cv2((tx, ty), angle, scale)
    # warped = cv2.warpAffine(image, A2, (width, height))

def get_affine_cv2(translationXY, rotation, scale, centerXY):
    """translationXY is touple (x,y), rotation is angle counter clockwise, no scale is 0, centerXY of rotation is a touple (cx, cy) 
    https://towardsdatascience.com/image-geometric-transformation-in-numpy-and-opencv-936f5cd1d315"""
    sin_theta = numpy.sin(rotation)
    cos_theta = numpy.cos(rotation)
    
    a_11 = scale * cos_theta
    a_21 = -sin_theta
    
    a_12 = sin_theta
    a_22 = scale * cos_theta
        
    a_13 = (1 - cos_theta)*centerXY[0] - sin_theta*centerXY[1] + translationXY[0]
    a_23 = sin_theta*centerXY[0] + (1 - cos_theta)*centerXY[1] + translationXY[1]
    return numpy.array([[a_11, a_12, a_13],
                     [a_21, a_22, a_23]])

def find_parameters_for_combined_top_view(imgBack, imgLeft, imgFront, imgRight, Back_position=None, Left_position=None, Front_position=None, Right_position=None):
    def emptyFunction(newValue):
        pass
    
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 1000, 500)
    if Back_position is None or Left_position is None or Front_position is None or Right_position is None:
        cv2.createTrackbar("vertical_offset_for_parallel", "TrackBars", int(imgFront.shape[1]/2),int(imgFront.shape[1]),emptyFunction)
        cv2.createTrackbar("horizontal_offset_for_parallel", "TrackBars", int(imgLeft.shape[1]/4),int(imgLeft.shape[1]/2),emptyFunction)
        cv2.createTrackbar("vertical_offset_for_perpendicular", "TrackBars", int(imgLeft.shape[0]/2),int(imgLeft.shape[0]),emptyFunction)
        cv2.createTrackbar("horizontal_offset_for_perpendicular", "TrackBars", int(imgFront.shape[0]/2),int(imgFront.shape[0]),emptyFunction)
        
        cv2.createTrackbar("vertical_scale_for_parallel", "TrackBars", 49,99,emptyFunction)
        cv2.createTrackbar("horizontal_scale_for_parallel", "TrackBars", 49,99,emptyFunction)
        cv2.createTrackbar("vertical_scale_for_perpendicular", "TrackBars", 49,99,emptyFunction)
        cv2.createTrackbar("horizontal_scale_for_perpendicular", "TrackBars", 49,99,emptyFunction)
        
    else:
        cv2.createTrackbar("vertical_offset_for_parallel", "TrackBars", int(Back_position[0]),int(imgFront.shape[1]),emptyFunction)
        cv2.createTrackbar("horizontal_offset_for_parallel", "TrackBars", int(Back_position[1]),int(imgLeft.shape[1]/2),emptyFunction)
        cv2.createTrackbar("vertical_offset_for_perpendicular", "TrackBars", int(Right_position[0]),int(imgLeft.shape[0]),emptyFunction)
        cv2.createTrackbar("horizontal_offset_for_perpendicular", "TrackBars", int(Right_position[1]),int(imgFront.shape[0]),emptyFunction)
        
        cv2.createTrackbar("vertical_scale_for_parallel", "TrackBars", int(Back_position[6]),99,emptyFunction)
        cv2.createTrackbar("horizontal_scale_for_parallel", "TrackBars", int(Back_position[7]),99,emptyFunction)
        cv2.createTrackbar("vertical_scale_for_perpendicular", "TrackBars", int(Right_position[6]),99,emptyFunction)
        cv2.createTrackbar("horizontal_scale_for_perpendicular", "TrackBars", int(Right_position[7]),99,emptyFunction)
    
    while True:
        #adjust parameters
        vertical_offset_for_parallel = cv2.getTrackbarPos("vertical_offset_for_parallel", "TrackBars") - int(imgFront.shape[1]/2)
        horizontal_offset_for_parallel = cv2.getTrackbarPos("horizontal_offset_for_parallel", "TrackBars") - int(imgLeft.shape[1]/4)
        vertical_offset_for_perpendicular = cv2.getTrackbarPos("vertical_offset_for_perpendicular", "TrackBars") - int(imgLeft.shape[0]/2)
        horizontal_offset_for_perpendicular = cv2.getTrackbarPos("horizontal_offset_for_perpendicular", "TrackBars") - int(imgFront.shape[0]/2)
        
        vertical_scale_for_parallel = (cv2.getTrackbarPos("vertical_scale_for_parallel", "TrackBars") + 1)/50
        horizontal_scale_for_parallel = (cv2.getTrackbarPos("horizontal_scale_for_parallel", "TrackBars") + 1)/50
        vertical_scale_for_perpendicular = (cv2.getTrackbarPos("vertical_scale_for_perpendicular", "TrackBars") + 1)/50
        horizontal_scale_for_perpendicular = (cv2.getTrackbarPos("horizontal_scale_for_perpendicular", "TrackBars") + 1)/50
        
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
        combined_top_view = combine_top_views(imgBack, imgLeft, imgFront, imgRight, Back_position, Left_position, Front_position, Right_position)
        cv2.imshow("combined_top_view", cv2.resize(combined_top_view, (0, 0), None, 0.8, 0.8))
        
        keyInput = cv2.waitKey(33)
        if keyInput==27:    # Esc key to stop
            break
        elif keyInput==-1:  # normally -1 returned,so don't print it
            continue
        else:
            print(keyInput) # else print its value
        
    cv2.destroyWindow("TrackBars")
    cv2.destroyWindow("combined_top_view")
    
    
    return Back_position, Left_position, Front_position, Right_position

def combine_top_views(imgBack, imgLeft, imgFront, imgRight, Back_position=None, Left_position=None, Front_position=None, Right_position=None):
    #Position is a list containing x_position, y_position, z_position,
    #                              camera_pitch, camera_yaw, camera_roll,
    #                              image_scale_x, image_scale_y
    # top right is begining x> y\/
    
    #rotation // rotation then scale would give better quallity, but not speed
    #obecnie sa idealnie ustawione, dla testow
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
    
    #scale
    imgBack_rotated = cv2.resize(imgBack_rotated, (0, 0), None, Back_position[6], Back_position[7])
    imgLeft_rotated = cv2.resize(imgLeft_rotated, (0, 0), None, Left_position[6], Left_position[7])
    imgFront = cv2.resize(imgFront, (0, 0), None, Front_position[6], Front_position[7])
    imgRight_rotated = cv2.resize(imgRight_rotated, (0, 0), None, Right_position[6], Right_position[7])
    
    #calculate imgCombined size
    imgCombined_height = max(imgFront.shape[0] - Front_position[1] + imgBack_rotated.shape[0] + Back_position[1], imgLeft_rotated.shape[0], imgRight_rotated.shape[0])
    imgCombined_width = max(imgRight_rotated.shape[1] + Right_position[0] + imgLeft_rotated.shape[1] - Left_position[0], imgFront.shape[1], imgBack_rotated.shape[1])
    center_x = int(imgCombined_width/2)
    center_y = int(imgCombined_height/2)
    
    
    #translation
    # M_Back = get_affine_cv2((Back_position[0], Back_position[1]+center_y), Back_position[4], Back_position[6])
    # M_Left = get_affine_cv2((Left_position[0], Left_position[1]), Left_position[4], Left_position[6])
    # M_Front = get_affine_cv2((Front_position[0], Front_position[1]), Front_position[4], Front_position[6])
    # M_Right = get_affine_cv2((Right_position[0]+center_x, Right_position[1]), Right_position[4], Right_position[6])
    
    # M_Back = get_translation2(Back_position[0], Back_position[1]+center_y)
    # M_Left = get_translation2(Left_position[0], Left_position[1])
    # M_Front = get_translation2(Front_position[0], Front_position[1])
    # M_Right = get_translation2(Right_position[0]+center_x, Right_position[1])
    
    # M_Back = get_translation2(center_x-int(imgBack_rotated.shape[1]/2)+Back_position[0], center_y-int(imgBack_rotated.shape[0]/2)+Back_position[1])
    M_Back = get_translation2(center_x-int(imgBack_rotated.shape[1]/2)+Back_position[0], center_y+Back_position[1])
    # M_Left = get_translation2(center_x-int(imgLeft_rotated.shape[1]/2)+Left_position[0], center_y-int(imgLeft_rotated.shape[0]/2)+Left_position[1])
    M_Left = get_translation2(center_x-int(imgLeft_rotated.shape[1])+Left_position[0], center_y-int(imgLeft_rotated.shape[0]/2)+Left_position[1])
    # M_Front = get_translation2(center_x-int(imgFront.shape[1]/2)+Front_position[0], center_y-int(imgFront.shape[0]/2)+Front_position[1])
    M_Front = get_translation2(center_x-int(imgFront.shape[1]/2)+Front_position[0], center_y-int(imgFront.shape[0])+Front_position[1])
    # M_Right = get_translation2(center_x-int(imgRight_rotated.shape[1]/2)+Right_position[0], center_y-int(imgRight_rotated.shape[0]/2)+Right_position[1])
    M_Right = get_translation2(center_x+Right_position[0], center_y-int(imgRight_rotated.shape[0]/2)+Right_position[1])
    
    warpedBack = cv2.warpAffine(imgBack_rotated, M_Back, (imgCombined_width, imgCombined_height))
    warpedLeft = cv2.warpAffine(imgLeft_rotated, M_Left, (imgCombined_width, imgCombined_height))
    warpedFront = cv2.warpAffine(imgFront, M_Front, (imgCombined_width, imgCombined_height))
    warpedRight = cv2.warpAffine(imgRight_rotated, M_Right, (imgCombined_width, imgCombined_height))
    
    imgCombined1 = numpy.where(warpedRight == 0, warpedLeft, warpedRight)
    imgCombined2 = numpy.where(warpedFront == 0, warpedBack, warpedFront)
    imgCombined0 = numpy.where(imgCombined2 == 0, imgCombined1, imgCombined2)
    
    # cv2.imshow("Wrap Affine Back", warpedBack)
    # cv2.imshow("Wrap Affine Left", warpedLeft)
    # cv2.imshow("Wrap Affine Front", warpedFront)
    # cv2.imshow("Wrap Affine Right", warpedRight)
    # cv2.imshow("img combined", imgCombined0)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # sys.exit()
    
    return imgCombined0

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
    return xmap, ymap


def stack_two_images_with_offsets(img1,img2, offset1, offset2):
    # horizontalSize = img1.shape[0] # lets assume height is the same and dont changes
    # verticalSize = img1.shape[1] - offset1 + img2.shape[1] - offset2
    # depthSize = img1.shape[2]
    # stackedImage = numpy.zeros((horizontalSize, verticalSize), numpy.uint8)
    stackedImage = numpy.concatenate((img1[:,0:img1.shape[1]-offset1,:], img2[:,offset2:img2.shape[1],:]), axis=1)
    return stackedImage

def find_parameters_for_two_image_stack(img1, img2):
    def emptyFunction(newValue):
        pass
    
    # img1Copy = img1.copy()
    # img2Copy = img2.copy()
    
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 900, 140)
    cv2.createTrackbar("Left offset", "TrackBars", 0,img1.shape[1],emptyFunction)
    cv2.createTrackbar("Right offset", "TrackBars", 0,img2.shape[1],emptyFunction)
        
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
        if keyInput==27:    # Esc key to stop
            break
        elif keyInput==-1:  # normally -1 returned,so don't print it
            continue
        else:
            print(keyInput) # else print its value
        
    cv2.destroyWindow("TrackBars")
    cv2.destroyWindow("stackedTwoImages")
    
    return offset1, offset2

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    # https://stackoverflow.com/a/20355545
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
    return result

def stitch_two_images_using_ORB(img1,img2): # there is also brisk algorithm in free cv2 library
    # Compute key points and feature descriptors
    # descriptor = cv2.ORB_create()
    descriptor = cv2.BRISK_create()
    (kp1, des1) = descriptor.detectAndCompute(img1, None)
    (kp2, des2) = descriptor.detectAndCompute(img2, None)
    # Using bruteforce, FLANN is faster, but can not be as accurate
    # https://stackoverflow.com/questions/10610966/difference-between-bfmatcher-and-flannbasedmatcher
    # https://docs.opencv2.org/3.4.1/dc/dc3/tutorial_py_matcher.html
    
    
    # BFMatcher with default params
    # option 1
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2, k=2)
    
    # option 2
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # matches = bf.knnMatch(des1,des2, k=2)
    
    # # Apply ratio test
    # good_matches = []
    # allmatchesm = []
    # for m,n in matches:
    #     allmatchesm.append(m)
    #     if m.distance < 0.75*n.distance:
    #         good_matches.append(m)
            
        
    # def drawMatches(img1, kp1, img2, kp2, matches):
    #     """
    #     My own implementation of cv2.drawMatches as OpenCV 2.4.9
    #     does not have this function available but it's supported in
    #     OpenCV 3.0.0
    
    #     This function takes in two images with their associated 
    #     keypoints, as well as a list of DMatch data structure (matches) 
    #     that contains which keypoints matched in which images.
    
    #     An image will be produced where a montage is shown with
    #     the first image followed by the second image beside it.
    
    #     Keypoints are delineated with circles, while lines are connected
    #     between matching keypoints.
    
    #     img1,img2 - Grayscale images
    #     kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
    #               detection algorithms
    #     matches - A list of matches of corresponding keypoints through any
    #               OpenCV keypoint matching algorithm
    #     """
    
    #     # Create a new output image that concatenates the two images together
    #     # (a.k.a) a montage
    #     rows1 = img1.shape[0]
    #     cols1 = img1.shape[1]
    #     rows2 = img2.shape[0]
    #     cols2 = img2.shape[1]
    
    #     # Create the output image
    #     # The rows of the output are the largest between the two images
    #     # and the columns are simply the sum of the two together
    #     # The intent is to make this a colour image, so make this 3 channels
    #     out = numpy.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    
    #     # Place the first image to the left
    #     out[:rows1,:cols1] = numpy.dstack([img1, img1, img1])
    
    #     # Place the next image to the right of it
    #     out[:rows2,cols1:] = numpy.dstack([img2, img2, img2])
    
    #     # For each pair of points we have between both images
    #     # draw circles, then connect a line between them
    #     for mat in matches:
    
    #         # Get the matching keypoints for each of the images
    #         img1_idx = mat.queryIdx
    #         img2_idx = mat.trainIdx
    
    #         # x - columns
    #         # y - rows
    #         (x1,y1) = kp1[img1_idx].pt
    #         (x2,y2) = kp2[img2_idx].pt
    
    #         # Draw a small circle at both co-ordinates
    #         # radius 4
    #         # colour blue
    #         # thickness = 1
    #         cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
    #         cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
    
    #         # Draw a line in between the two points
    #         # thickness = 1
    #         # colour blue
    #         cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255,0,0), 1)
    
    
    #     # Show the image
    #     cv2.imshow('Matched Features', out)
    #     cv2.waitKey(0)
    #     cv2.destroyWindow('Matched Features')
    
    #     # Also return the image if you'd like a copy
    #     return out

    # # cv2.drawMatchesKnn expects list of lists as matches.
    # gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    # # drawMatches(gray1,kp1,gray2,kp2,allmatchesm) #draw all
    # drawMatches(gray1,kp1,gray2,kp2,good_matches) # draw only best
    
    # MIN_MATCH_COUNT = 10 # moze mniej moze wiecej
    # if len(good_matches)>MIN_MATCH_COUNT:
    #     src_pts = numpy.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    #     dst_pts = numpy.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        
    #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    #     matchesMask = mask.ravel().tolist()
        
    #     h,w,d = img1.shape
    #     pts = numpy.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #     dst = cv2.perspectiveTransform(pts,M)
        
    #     img2 = cv2.polylines(img2,[numpy.int32(dst)],True,255,3, cv2.LINE_AA)
        
    #     cv2.imshow("CoDo", img2)
    #     cv2.waitKey(0)
    #     cv2.destroyWindow("CoDo")
    # else:
    #     print( "Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT) )
    #     matchesMask = None
        
    
    #option 3
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # img1 = query image, img2 = train image
    good_key_point_in_img1 = []
    good_key_point_in_img2 = []
    for match in matches[:100]:
        good_key_point_in_img1.append((kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1]))
        good_key_point_in_img2.append((kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1]))
    
    good_key_point_in_img1 = numpy.array(good_key_point_in_img1, dtype=numpy.float32)
    good_key_point_in_img2 = numpy.array(good_key_point_in_img2, dtype=numpy.float32)

    # # Draw first 10 matches from key points.
    # img4 = numpy.concatenate((img1, img2), axis=1)
    # for i in range(10):
    #     point1 = int(good_key_point_in_img1[i,0]), int(good_key_point_in_img1[i,1])
    #     point2 = int(good_key_point_in_img2[i,0]), int(good_key_point_in_img2[i,1])
    #     cv2.circle(img4, (point1[0],point1[1]), radius=10, color=(0, 0, 255), thickness=-1)
    #     cv2.circle(img4, (point2[0]+img1.shape[1],point2[1]), radius=10, color=(0, 0, 255), thickness=-1)
    #     cv2.line(img4, (point1[0],point1[1]), (point2[0]+img1.shape[1],point2[1]), (0, 255, 0), thickness=5)
    # cv2.imshow("Best points", cv2.resize(img4, (0, 0), None, 0.5, 0.5))
    # cv2.waitKey(0)
    # cv2.destroyWindow("Best points")
    # sys.exit()
    
    M, mask = cv2.findHomography(good_key_point_in_img2, good_key_point_in_img1, cv2.RANSAC,5.0)

    # print("Mask")
    # print(mask)
    # print("M")
    # print(M)
    # cv2.imshow("Matched Features", img3)
    # cv2.waitKey(0)
    # cv2.destroyWindow("Matched Features")
    
    #Dzialajaca 50/50 xD
    # # Apply panorama correction
    # width = img2.shape[1] + img1.shape[1]
    # height = img2.shape[0] + img1.shape[0]
    
    # result = cv2.warpPerspective(img1, M, (width, height))
    # # result = cv2.perspectiveTransform(img2,M)
    # result[0:img2.shape[0], 0:img2.shape[1]] = img2
    
    # Poprawna metoda
    result = warpTwoImages(img1, img2, M)
    
    # cv2.imshow("Panorama", cv2.resize(result, (0, 0), None, 0.5, 0.5))
    # cv2.waitKey(0)
    # cv2.destroyWindow("Panorama")
    return result


def main():
    # calibrationDirectory = 'dataset5/*.jpg'
    # calibrationDirectory = glob.glob(calibrationDirectory)
    # calibrationDirectory = only_valid_images_for_calibration
    # K,D,DIM = getCameraParameters_using_omnidirectional(calibrationDirectory)
    # K,D,DIM = getCameraParameters(calibrationDirectory)
    K = numpy.array([[219.85077387813544, 0.0, 321.8468539428703], [0.0, 219.81115217715458, 321.26199300586325], [0.0, 0.0, 1.0]])
    D = numpy.array([[-0.02236163741176025], [-0.01566355538478192], [0.0066695817100666304], [-0.0009867103996664935]])
    DIM =(640, 640)
    
    # testimg_bird_eye = cv2.imread("dataset5/0023.jpg")
    
    # undistorted_testimg3 = undistort(testimg_bird_eye, K, D, DIM)
    # undistorted_testimg2 = undistort2(testimg_bird_eye, K, D, DIM, balance=0.5, dim2=(840,840), dim3=(840,840))
    # undistorted_testimg = undistort3(testimg_bird_eye, K, D, DIM, balance=0.5, dim2=(840,840), dim3=(840,840))
    # undistorted_testimg4 = undistort4(testimg_bird_eye, K, D, DIM, balance=0.5)
    # # cv2.imshow("Undistorted Test", numpy.concatenate((undistorted_testimg2,undistorted_testimg), axis=1))
    # cv2.imshow("Undistorted Test using 1", undistorted_testimg3)
    # cv2.imshow("Undistorted Test using 2", undistorted_testimg2)
    # cv2.imshow("Undistorted Test using 3", undistorted_testimg)
    # cv2.imshow("Undistorted Test using 4", undistorted_testimg4)
    
    # # make_top_view(undistorted_testimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # sys.exit()
    
    #cap = cv2.VideoCapture("Front_120_2500000-0231.mp4")
    capBack = cv2.VideoCapture("260-290mp4/Back_0260-0290.mp4")
    capLeft = cv2.VideoCapture("260-290mp4/Left_0260-0290.mp4")
    capFront = cv2.VideoCapture("260-290mp4/Front_0260-0290.mp4")
    capRight = cv2.VideoCapture("260-290mp4/Right_0260-0290.mp4")
    
    
    # img_path = '0004.jpg'
    # img = cv2.imread(img_path)
    # undistort(img,K,D,DIM)
    frame_counter = 0
    
    
    #first read
    successBack, imgBack = capBack.read()
    successLeft, imgLeft = capLeft.read()
    successFront, imgFront = capFront.read()
    successRight, imgRight = capRight.read()
    
    if successBack is True and successLeft is True and successFront is True and successRight is True:
        success = True
        frame_counter += 1
    else:
        success = False
        cv2.destroyAllWindows()
        return
    #end first read
    
    if ONLY_TOP_VIEW is True:
        #unwrap images using fisheye calibration
        imgBack_unwarped0 = undistort3(imgBack,K,D,DIM)
        imgLeft_unwarped0 = undistort3(imgLeft,K,D,DIM)
        imgFront_unwarped0 = undistort3(imgFront,K,D,DIM)
        imgRight_unwarped0 = undistort3(imgRight,K,D,DIM)
        img3 = numpy.concatenate((imgBack_unwarped0, imgLeft_unwarped0, imgFront_unwarped0, imgRight_unwarped0), axis=1)
    
        # shrinking_parameter, crop_top, crop_bottom = find_parameters_to_make_top_view(imgBack_unwarped0)
        shrinking_parameter = 300
        crop_top = 340
        crop_bottom = 0
        
        imgBack_topview = make_top_view(imgBack_unwarped0, shrinking_parameter=shrinking_parameter, crop_top=crop_top, crop_bottom=crop_bottom)
        imgLeft_topview = make_top_view(imgLeft_unwarped0, shrinking_parameter=shrinking_parameter, crop_top=crop_top, crop_bottom=crop_bottom)
        imgFront_topview = make_top_view(imgFront_unwarped0, shrinking_parameter=shrinking_parameter, crop_top=crop_top, crop_bottom=crop_bottom)
        imgRight_topview = make_top_view(imgRight_unwarped0, shrinking_parameter=shrinking_parameter, crop_top=crop_top, crop_bottom=crop_bottom)
        img4 = numpy.concatenate((imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview), axis=1)
        cv2.imshow("Unwarped by undistort in line", cv2.resize(img3, (0, 0), None, 0.5, 0.5))
        cv2.imshow("Top view", cv2.resize(img4, (0, 0), None, 0.5, 0.5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # sys.exit()
        
        # Back_position = [0, 8, 0, 0, 0, 0, 1.38, 0.56]
        # Left_position = [0, 0, 0, 0, 0, 0, 0.54, 1.56]
        # Front_position = [0, -8, 0, 0, 0, 0, 1.38, 0.56]
        # Right_position = [0, 0, 0, 0, 0, 0, 0.54, 1.56]
        
        vertical_offset_for_parallel = 320
        horizontal_offset_for_parallel = 160
        vertical_offset_for_perpendicular = 150
        horizontal_offset_for_perpendicular = 145
        
        vertical_scale_for_parallel = 75
        horizontal_scale_for_parallel = 30
        vertical_scale_for_perpendicular = 29
        horizontal_scale_for_perpendicular = 76
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
        Back_position, Left_position, Front_position, Right_position = find_parameters_for_combined_top_view(imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview, Back_position, Left_position, Front_position, Right_position)
        # print("Back_position")
        # print(Back_position)
        # print("Left_position")
        # print(Left_position)
        # print("Front_position")
        # print(Front_position)
        # print("Right_position")
        # print(Right_position)
        
        
        combined_top_view = combine_top_views(imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview, Back_position, Left_position, Front_position, Right_position)
        #cv2.imshow("Top View image", combined_top_view)
    else:
        
        # #unwarp images using projection
        W_remap = 720
        H = 640
        FOV = 180
        xmap, ymap = buildmap(Ws=W_remap, Hs=H, Wd=640, Hd=640, fov=FOV)
        imgBack_unwarped = cv2.remap(imgBack, xmap, ymap, cv2.INTER_LINEAR)
        imgLeft_unwarped = cv2.remap(imgLeft, xmap, ymap, cv2.INTER_LINEAR)
        imgFront_unwarped = cv2.remap(imgFront, xmap, ymap, cv2.INTER_LINEAR)
        imgRight_unwarped = cv2.remap(imgRight, xmap, ymap, cv2.INTER_LINEAR)
        
        img4 = numpy.concatenate((imgBack_unwarped, imgLeft_unwarped, imgFront_unwarped, imgRight_unwarped), axis=1)
        cv2.imshow("Unwarped by undistort in line", cv2.resize(img3, (0, 0), None, 0.5, 0.5))
        cv2.imshow("Unwarped in line", cv2.resize(img4, (0, 0), None, 0.5, 0.5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # sys.exit()
        #stitch unwarped images
        # vertical_stitching_offset = 40
        vertical_stitching_offset = 30
        horizontal_stitching_offset = 80
        # Zrobic to tak, ze funkcja dostaje mniejsze ROI do poszukiwania cech wspolnych, a pozniej przesunac punkty, bedzie szbysze i dokladniejsze
        #opcja z doklejaniem
        # stitched_BL = stitch_two_images_using_ORB(imgBack_unwarped[horizontal_stitching_offset:H-horizontal_stitching_offset, vertical_stitching_offset:W_remap-vertical_stitching_offset, :],imgLeft_unwarped[horizontal_stitching_offset:H-horizontal_stitching_offset, vertical_stitching_offset:W_remap-vertical_stitching_offset, :])
        # stitched_BLF = stitch_two_images_using_ORB(stitched_BL,imgFront_unwarped[horizontal_stitching_offset:H-horizontal_stitching_offset, vertical_stitching_offset:W_remap-vertical_stitching_offset, :])
        # stitched_BLFR = stitch_two_images_using_ORB(stitched_BLF,imgRight_unwarped[horizontal_stitching_offset:H-horizontal_stitching_offset, vertical_stitching_offset:W_remap-vertical_stitching_offset, :])
        # print("BLFR\n")
        # print(stitched_BLFR.shape)
        #opcja rownolegla
        stitched_BL = stitch_two_images_using_ORB(imgBack_unwarped[horizontal_stitching_offset:H-horizontal_stitching_offset, vertical_stitching_offset:W_remap-vertical_stitching_offset, :],imgLeft_unwarped[horizontal_stitching_offset:H-horizontal_stitching_offset, vertical_stitching_offset:W_remap-vertical_stitching_offset, :])
        stitched_FR = stitch_two_images_using_ORB(imgRight_unwarped[horizontal_stitching_offset:H-horizontal_stitching_offset, vertical_stitching_offset:W_remap-vertical_stitching_offset, :],imgFront_unwarped[horizontal_stitching_offset:H-horizontal_stitching_offset, vertical_stitching_offset:W_remap-vertical_stitching_offset, :])
        stitched_BLRF = stitch_two_images_using_ORB(stitched_BL,stitched_FR)
        print("BLRF\n")
        print(stitched_BLRF.shape)
        
        # crop image - czy to ma sens? - raczej nie, zobaczymy jak to będzie dzialac dalej, teraz gupie
        # moze wypadaloby zrobic projekcje z "equrectangular"/"obround" do "rectangular" przedtym
        height_in_stiched = numpy.nonzero(stitched_BLRF[:,int(stitched_BLRF.shape[1]/7),:][:,1])[0]
        width_in_stiched = numpy.nonzero(stitched_BLRF[int(stitched_BLRF.shape[0]/2),:,:][:,1])[0]
        croped_stiched_BLRF = stitched_BLRF[min(height_in_stiched):max(height_in_stiched), min(width_in_stiched):max(width_in_stiched),:]

        # cv2.imshow("BLFR",stitched_BLFR)
        cv2.imshow("BLRF",stitched_BLRF)
        cv2.imshow("Croped BLRF",croped_stiched_BLRF)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # sys.exit()
        
        #find parameters
        offsetBackLeft1, offsetBackLeft2 = find_parameters_for_two_image_stack(imgBack,imgLeft)
        stacked_back_left = stack_two_images_with_offsets(imgBack,imgLeft,offsetBackLeft1,offsetBackLeft2)
        
        offsetBackLeftFront1, offsetBackLeftFront2 = find_parameters_for_two_image_stack(stacked_back_left,imgFront)
        stacked_back_left_front = stack_two_images_with_offsets(stacked_back_left,imgFront,offsetBackLeftFront1,offsetBackLeftFront2)
        
        offsetBackLeftFrontRight1, offsetBackLeftFrontRight2 = find_parameters_for_two_image_stack(stacked_back_left_front,imgRight)
        stacked_back_left_front_right = stack_two_images_with_offsets(stacked_back_left_front,imgRight,offsetBackLeftFrontRight1,offsetBackLeftFrontRight2)
        
        #end find parameters
    
    while True:
        successBack, imgBack = capBack.read()
        successLeft, imgLeft = capLeft.read()
        successFront, imgFront = capFront.read()
        successRight, imgRight = capRight.read()
        
        if successBack is True and successLeft is True and successFront is True and successRight is True:
            success = True
            frame_counter += 1
        else:
            success = False
    
        if cv2.waitKey(1) & 0xFF == ord('q') or success is False:
            break
        else:
            
            if ONLY_TOP_VIEW is True:
                #only top view
                
                #unwrap images using fisheye calibration
                imgBack_unwarped0 = undistort3(imgBack,K,D,DIM)
                imgLeft_unwarped0 = undistort3(imgLeft,K,D,DIM)
                imgFront_unwarped0 = undistort3(imgFront,K,D,DIM)
                imgRight_unwarped0 = undistort3(imgRight,K,D,DIM)
                #make topview
                imgBack_topview = make_top_view(imgBack_unwarped0, shrinking_parameter=shrinking_parameter, crop_top=crop_top, crop_bottom=crop_bottom)
                imgLeft_topview = make_top_view(imgLeft_unwarped0, shrinking_parameter=shrinking_parameter, crop_top=crop_top, crop_bottom=crop_bottom)
                imgFront_topview = make_top_view(imgFront_unwarped0, shrinking_parameter=shrinking_parameter, crop_top=crop_top, crop_bottom=crop_bottom)
                imgRight_topview = make_top_view(imgRight_unwarped0, shrinking_parameter=shrinking_parameter, crop_top=crop_top, crop_bottom=crop_bottom)
                combined_top_view = combine_top_views(imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview, Back_position, Left_position, Front_position, Right_position)
                cv2.imshow("Top View image", combined_top_view)
    
            else:
                #stiching and equirectangular github
                # undistortedBack = undistort(imgBack,K,D,DIM)
                imgBackGRAY = cv2.cvtColor(imgBack,cv2.COLOR_BGR2GRAY)
                
                # undistortedLeft = undistort(imgLeft,K,D,DIM)
                imgLeftGRAY = cv2.cvtColor(imgLeft,cv2.COLOR_BGR2GRAY)
                
                # undistortedFront = undistort(imgFront,K,D,DIM)
                imgFrontGRAY = cv2.cvtColor(imgFront,cv2.COLOR_BGR2GRAY)
                
                # undistortedRight = undistort(imgRight,K,D,DIM)
                imgRightGRAY = cv2.cvtColor(imgRight,cv2.COLOR_BGR2GRAY)
                
                
                stacked_back_left = stack_two_images_with_offsets(imgBack,imgLeft,offsetBackLeft1,offsetBackLeft2)
                stacked_back_left_front = stack_two_images_with_offsets(stacked_back_left,imgFront,offsetBackLeftFront1,offsetBackLeftFront2)
                stacked_back_left_front_right = stack_two_images_with_offsets(stacked_back_left_front,imgRight,offsetBackLeftFrontRight1,offsetBackLeftFrontRight2)
                cv2.imshow("stacked_back_left_front_right", stacked_back_left_front_right)
                
                
                stacked = numpy.hstack((cv2.resize(imgBackGRAY, (0, 0), None, 0.5, 0.5), \
                             cv2.resize(imgLeftGRAY, (0, 0), None, 0.5, 0.5), \
                                 cv2.resize(imgFrontGRAY, (0, 0), None, 0.5, 0.5), \
                                     cv2.resize(imgRightGRAY, (0, 0), None, 0.5, 0.5) \
                             ))
                cv2.imshow("stacked", stacked)
            
            # stackedTwo = stack_two_images_with_offsets(imgLeft,imgFront,100,100)
            # cv2.imshow("stackedTwo", stackedTwo)
        
            #loop video
            if frame_counter == capBack.get(cv2.CAP_PROP_FRAME_COUNT) or \
                frame_counter == capLeft.get(cv2.CAP_PROP_FRAME_COUNT) or \
                    frame_counter == capFront.get(cv2.CAP_PROP_FRAME_COUNT) or \
                        frame_counter == capRight.get(cv2.CAP_PROP_FRAME_COUNT):
                frame_counter = 0
                capBack.set(cv2.CAP_PROP_POS_FRAMES, 0)
                capLeft.set(cv2.CAP_PROP_POS_FRAMES, 0)
                capFront.set(cv2.CAP_PROP_POS_FRAMES, 0)
                capRight.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            #make it visible for human eye
            time.sleep(0.2)
        

    # cv2.destroyWindow("UndistortedBack")
    # cv2.destroyWindow("UndistortedLeft")
    # cv2.destroyWindow("UndistortedFront")
    # cv2.destroyWindow("UndistortedRight")
    cv2.destroyWindow("stacked")
    cv2.destroyWindow("stacked_back_left_front_right")
    
    
    # offsetBackLeft1, offsetBackLeft2 = find_parameters_for_two_image_stack(imgBack,imgLeft)
    # stacked_back_left = stack_two_images_with_offsets(imgBack,imgLeft,offsetBackLeft1,offsetBackLeft2)
    
    cv2.destroyAllWindows()
    capBack.release()
    capLeft.release()
    capFront.release()
    capRight.release()



only_valid_images_for_calibration = []
only_valid_images_for_calibration.append("dataset5/0009.jpg")
only_valid_images_for_calibration.append("dataset5/0010.jpg")
only_valid_images_for_calibration.append("dataset5/0023.jpg")
only_valid_images_for_calibration.append("dataset5/0032.jpg")
only_valid_images_for_calibration.append("dataset5/0058.jpg")
only_valid_images_for_calibration.append("dataset5/0067.jpg")

ONLY_TOP_VIEW = True
main()


# #test laczenia zdjec
# testowe0 = cv2.imread("zdjTestPanoramy2.png")
# testowe1 = cv2.imread("zdjTestPanoramy3.png")

# stitch_two_images_using_ORB(cv2.resize(testowe0, (0, 0), None, 0.5, 0.5), cv2.resize(testowe1, (0, 0), None, 0.5, 0.5))

# #test przeksztalcenia aficznego
# img = cv2.imread("zdjTestPanoramy2.png")
# # A2 = get_affine_cv2((0, 0), 180*numpy.pi/180, 0.5, (int(img.shape[1]/2),int(img.shape[0]/2)))
# A2 = get_affine_cv((0, 0), 1, 0.5)
# warped = cv2.warpAffine(img, A2, (img.shape[:2][::-1]))

# cv2.imshow("Source", img)
# cv2.imshow("Result", warped)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #test top view
# imgBack = cv2.imread("System_Calibration-3/0002.jpg")
# imgLeft = cv2.imread("System_Calibration-3/0003.jpg")
# imgFront = cv2.imread("System_Calibration-3/0000.jpg")
# imgRight = cv2.imread("System_Calibration-3/0001.jpg")

# K = numpy.array([[219.85077387813544, 0.0, 321.8468539428703], [0.0, 219.81115217715458, 321.26199300586325], [0.0, 0.0, 1.0]])
# D = numpy.array([[-0.02236163741176025], [-0.01566355538478192], [0.0066695817100666304], [-0.0009867103996664935]])
# DIM =(640, 640)
# imgBack_unwarped0 = undistort3(imgBack,K,D,DIM)
# imgLeft_unwarped0 = undistort3(imgLeft,K,D,DIM)
# imgFront_unwarped0 = undistort3(imgFront,K,D,DIM)
# imgRight_unwarped0 = undistort3(imgRight,K,D,DIM)
# #do ulicznego?
# # shrinking_parameter = 300
# # crop_top = 340
# # crop_bottom = 0
# #do calibrate system
# shrinking_parameter = 300
# crop_top = 331
# crop_bottom = 140
# # shrinking_parameter, crop_top, crop_bottom = find_parameters_to_make_top_view(imgFront_unwarped0)
# imgBack_topview = make_top_view(imgBack_unwarped0, shrinking_parameter=shrinking_parameter, crop_top=crop_top, crop_bottom=crop_bottom)
# imgLeft_topview = make_top_view(imgLeft_unwarped0, shrinking_parameter=shrinking_parameter, crop_top=crop_top, crop_bottom=crop_bottom)
# imgFront_topview = make_top_view(imgFront_unwarped0, shrinking_parameter=shrinking_parameter, crop_top=crop_top, crop_bottom=crop_bottom)
# imgRight_topview = make_top_view(imgRight_unwarped0, shrinking_parameter=shrinking_parameter, crop_top=crop_top, crop_bottom=crop_bottom)

# #do ulicznego?
# # Back_position = [0, 8, 0, 0, 0, 0, 1.38, 0.56]
# # Left_position = [0, 0, 0, 0, 0, 0, 0.54, 1.56]
# # Front_position = [0, -8, 0, 0, 0, 0, 1.38, 0.56]
# # Right_position = [0, 0, 0, 0, 0, 0, 0.54, 1.56]

# #do calibrate system
# vertical_offset_for_parallel = 314
# horizontal_offset_for_parallel = 224
# vertical_offset_for_perpendicular = 117
# horizontal_offset_for_perpendicular = 87

# vertical_scale_for_parallel = 61
# horizontal_scale_for_parallel = 45
# vertical_scale_for_perpendicular = 40
# horizontal_scale_for_perpendicular = 59
# Back_position = [vertical_offset_for_parallel,  horizontal_offset_for_parallel,  0,  \
#           0,  0,  0,     \
#               vertical_scale_for_parallel, horizontal_scale_for_parallel]
# Left_position = [-vertical_offset_for_perpendicular,  horizontal_offset_for_perpendicular,  0,  \
#           0,  0,  0,     \
#               vertical_scale_for_perpendicular, horizontal_scale_for_perpendicular]
# Front_position = [vertical_offset_for_parallel,  -horizontal_offset_for_parallel,  0,  \
#           0,  0,  0,     \
#               vertical_scale_for_parallel, horizontal_scale_for_parallel]
# Right_position = [vertical_offset_for_perpendicular,  horizontal_offset_for_perpendicular,  0,  \
#           0,  0,  0,     \
#               vertical_scale_for_perpendicular, horizontal_scale_for_perpendicular]
# # Back_position = [0, 8, 0, 0, 0, 0, 1.38, 0.56]
# # Left_position = [0, 0, 0, 0, 0, 0, 0.54, 1.56]
# # Front_position = [0, -8, 0, 0, 0, 0, 1.38, 0.56]
# # Right_position = [0, 0, 0, 0, 0, 0, 0.54, 1.56]
# Back_position, Left_position, Front_position, Right_position = find_parameters_for_combined_top_view(imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview, Back_position, Left_position, Front_position, Right_position)
# combined_top_view = combine_top_views(imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview, Back_position, Left_position, Front_position, Right_position)

# cv2.imshow("Result of top view test", cv2.resize(combined_top_view, (0, 0), None, 0.8, 0.8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()