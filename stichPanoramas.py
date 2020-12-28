# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:51:40 2020

@author: Dawid
"""
import cv2
assert int(cv2.__version__[0]) >= 3, 'The fisheye module requires opencv version >= 3.0.0'
import numpy
# import os
import glob
import time
# from matplotlib import pyplot as plt
import sys

 
def click_event(event, x, y, flags, params):
    """function to display the coordinates of the points clicked on the image"""
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)    
    if event==cv2.EVENT_RBUTTONDOWN:
        print(x, ' ', y)

  
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
            
        homography_matrix = cv2.getPerspectiveTransform(points_source, points_destination)
        if map_x is None or map_y is None:
            map_x, map_y = create_maps_using_homography_matrix(img, homography_matrix, invert_matrix=True)
        result = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, cv2.CV_32FC1)
        #result = cv2.warpPerspective(img, homography_matrix, img.shape[:2][::-1])
    else:
        # top view from parameters
        heigth_original, width = img.shape[:2]
        img2=img[crop_top:img.shape[0]-crop_bottom,:,:]
        heigth = heigth_original-crop_top-crop_bottom
        points_source = numpy.array([[0,0], [width,0], [0,heigth], [width,heigth]], dtype=numpy.float32)
        points_destination = numpy.array([[0,0], [width,0], [shrinking_parameter,heigth], [width-shrinking_parameter,heigth]], dtype=numpy.float32)
        
        homography_matrix = cv2.getPerspectiveTransform(points_source, points_destination)
        if map_x is None or map_y is None:
            map_x, map_y = create_maps_using_homography_matrix(img2, homography_matrix, invert_matrix=True)
        result = cv2.remap(img2, map_x, map_y, cv2.INTER_LINEAR, cv2.CV_32FC1)
        # result = cv2.warpPerspective(img2, homography_matrix, img2.shape[:2][::-1])
            
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

def find_undistortion_maps(img, K,D,DIM, balance=0.0, dim2=None, dim3=None):
    # balance not used
    # using undistort3() with undistort_with_maps()
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
    return (map1, map2)

def undistort_with_maps(img, maps):
    """https://stackoverflow.com/a/44009548"""
    undistorted_img = cv2.remap(img, maps[0], maps[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
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

def find_parameters_for_combine_top_view(imgBack, imgLeft, imgFront, imgRight, Back_position=None, Left_position=None, Front_position=None, Right_position=None):
    def emptyFunction(newValue):
        pass

    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 1000, 500)
    
    if Back_position is None or Left_position is None or Front_position is None or Right_position is None:
        cv2.createTrackbar("vertical_offset_for_parallel", "TrackBars", int(imgFront.shape[1]/2),int(imgFront.shape[1]*3/4),emptyFunction)
        cv2.createTrackbar("horizontal_offset_for_parallel", "TrackBars", int(imgLeft.shape[1]/4),int(imgLeft.shape[1]/2),emptyFunction)
        cv2.createTrackbar("vertical_offset_for_perpendicular", "TrackBars", int(imgLeft.shape[0]/2),int(imgLeft.shape[0]),emptyFunction)
        cv2.createTrackbar("horizontal_offset_for_perpendicular", "TrackBars", int(imgFront.shape[0]/2),int(imgFront.shape[0]*3/4),emptyFunction)
        
        # Scale can be fixed, that conversion was used to omit minimum 0, but there is cv2.setTrackbarMin
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
    
    mask_Back, mask_Left, mask_Front, mask_Right = combine_top_view(imgBack, imgLeft, imgFront, imgRight, Back_position, Left_position, Front_position, Right_position, return_masks = True, first_run = False)
    
    return Back_position, Left_position, Front_position, Right_position, mask_Back, mask_Left, mask_Front, mask_Right


def combine_top_view(imgBack, imgLeft, imgFront, imgRight, Back_position=None, Left_position=None, Front_position=None, Right_position=None, mask_Back=None, mask_Left=None, mask_Front=None, mask_Right=None, return_masks = False, first_run = False):
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
        
        # if False: # lepiej dac poprzednia metode, wiecej widac
        #     # calculate imgCombined size
        #     imgCombined_height = max(imgFront.shape[0] - Front_position[1] + imgBack_rotated.shape[0] + Back_position[1], imgLeft_rotated.shape[0], imgRight_rotated.shape[0])
        #     imgCombined_width = max(imgRight_rotated.shape[1] + Right_position[0] + imgLeft_rotated.shape[1] - Left_position[0], imgFront.shape[1], imgBack_rotated.shape[1])
        #     center_x = int(imgCombined_width/2)
        #     center_y = int(imgCombined_height/2)
            
        #     warpedBack = numpy.zeros((imgCombined_height, imgCombined_width, 3))
        #     warpedLeft = numpy.zeros((imgCombined_height, imgCombined_width, 3))
        #     warpedFront = numpy.zeros((imgCombined_height, imgCombined_width, 3))
        #     warpedRight = numpy.zeros((imgCombined_height, imgCombined_width, 3))
            
        #     # translation
        #     M_Back = (center_x-int(imgBack_rotated.shape[1]/2)+Back_position[0], center_y+Back_position[1])
        #     M_Left = (center_x-int(imgLeft_rotated.shape[1])+Left_position[0], center_y-int(imgLeft_rotated.shape[0]/2)+Left_position[1])
        #     M_Front = (center_x-int(imgFront.shape[1]/2)+Front_position[0], center_y-int(imgFront.shape[0])+Front_position[1])
        #     M_Right = (center_x+Right_position[0], center_y-int(imgRight_rotated.shape[0]/2)+Right_position[1])
            
            
        #     y_front = center_y - imgFront.shape[0] + Front_position[1] 
        #     y_back = center_y + imgBack_rotated.shape[0] + Back_position[1]
        #     x_right = center_x + imgRight_rotated.shape[1] + Right_position[0]
        #     x_left = center_x - imgLeft_rotated.shape[1] - Left_position[0]
            
        #     warpedBack[M_Back[1] : M_Back[1] + imgBack_rotated.shape[0], M_Back[0] + x_left : M_Back[0] + x_right, : ] = imgBack_rotated[:, x_left:x_right,:]
        #     warpedLeft[M_Left[1] + y_front : M_Left[1] + y_back, M_Left[0] : M_Left[0] + imgLeft_rotated.shape[1], : ] = imgLeft_rotated[y_front:y_back, :,:]
        #     warpedFront[M_Front[1] : M_Front[1] + imgFront.shape[0], M_Front[0] + x_left : M_Front[0] + x_right, : ] = imgFront[:, x_left:x_right,:]
        #     warpedRight[M_Right[1] + y_front : M_Right[1] + + y_back, M_Right[0] : M_Right[0] + imgRight_rotated.shape[1], : ] = imgRight_rotated[y_front:y_back, :,:]
        
        #     calculate_masks = True
        
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
        mask_Back = (warpedBack != 0)
        mask_Left = (warpedLeft != 0)
        mask_Front = (warpedFront != 0)
        mask_Right = (warpedRight != 0)

    
    # combine
    imgCombined = numpy.zeros((imgCombined_height, imgCombined_width, 3), dtype=numpy.uint8)
    imgCombined[mask_Left] = warpedLeft[mask_Left]
    imgCombined[mask_Right] = warpedRight[mask_Right]
    imgCombined[mask_Back] = warpedBack[mask_Back]
    imgCombined[mask_Front] = warpedFront[mask_Front]
    
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
        return mask_Back, mask_Left, mask_Front, mask_Right
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
    #stackedImage = numpy.concatenate((img1[:,0:img1.shape[1]-offset1,:], img2[:,offset2:img2.shape[1],:]), axis=1) # old method, more complex indexes
    stackedImage = numpy.concatenate((img1[:,:-offset1,:], img2[:,offset2:,:]), axis=1)
    return stackedImage

def find_parameters_for_two_image_stack(img1, img2, offset1=0, offset2=0):
    def emptyFunction(newValue):
        pass
    
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 900, 140)
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

def stitch_two_images_using_ORB(img1,img2, parameters = None): # there is also brisk algorithm in free cv2 library
    if parameters is None:
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
        result, parameters = warpTwoImages(img1, img2, H = M)
        
        # cv2.imshow("Panorama", cv2.resize(result, (0, 0), None, 0.5, 0.5))
        # cv2.waitKey(0)
        # cv2.destroyWindow("Panorama")
        
    else:
        result, parameters = warpTwoImages(img1, img2, parameters = parameters)
    
    return result, parameters


class Main():

    def __init__(self):
        self.K = None
        self.D = None
        self.DIM = None
        
        
        self.load_cameras()
        self.frame_counter = 0
        if MAKE_TOP_VIEW is True:
            self.calibrate_top_view()
        if MAKE_EQUIRECTANGULAR_PROJECTION is True:
            self.calibrate_equirectangular_projection()
            
        
    def load_cameras(self):
        self.capBack = cv2.VideoCapture("260-290mp4/Back_0260-0290.mp4")
        self.capLeft = cv2.VideoCapture("260-290mp4/Left_0260-0290.mp4")
        self.capFront = cv2.VideoCapture("260-290mp4/Front_0260-0290.mp4")
        self.capRight = cv2.VideoCapture("260-290mp4/Right_0260-0290.mp4")
        
        
    def calibrate_camera(self):
        """
        Calibrate camera
        """
        if USE_PREDEFINED_CAMERA_PARAMETERS is True:
            self.K = numpy.array([[219.85077387813544, 0.0, 321.8468539428703], [0.0, 219.81115217715458, 321.26199300586325], [0.0, 0.0, 1.0]])
            self.D = numpy.array([[-0.02236163741176025], [-0.01566355538478192], [0.0066695817100666304], [-0.0009867103996664935]])
            self.DIM =(640, 640)
        elif ONLY_VALID_IMAGES_FOR_CAMERA_CALIBRATION is True:
            calibrationDirectory = only_valid_images_for_calibration
            self.K, self.D, self.DIM = getCameraParameters_using_omnidirectional(calibrationDirectory)
            self.K, self.D, self.DIM = getCameraParameters(calibrationDirectory)
        else:
            calibrationDirectory = 'dataset5/*.jpg'
            calibrationDirectory = glob.glob(calibrationDirectory)
            self.K, self.D, self.DIM = getCameraParameters_using_omnidirectional(calibrationDirectory)
            self.K, self.D, self.DIM = getCameraParameters(calibrationDirectory)
        
        
        # Test undistort methods
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

        if USE_PREDEFINED_TOP_VIEW_PARAMETERS is True:
            # 1 version
            self.shrinking_parameter = 300
            self.crop_top = 340
            self.crop_bottom = 0
            # 3 version
            # shrinking_parameter = 290
            # crop_top = 350
            # crop_bottom = 0
            _, self.top_view_map_x, self.top_view_map_y = make_top_view(imgBack_unwarped0, shrinking_parameter=self.shrinking_parameter, crop_top=self.crop_top, crop_bottom=self.crop_bottom, return_maps=True)
        else:
            self.shrinking_parameter, self.crop_top, self.crop_bottom, self.top_view_map_x, self.top_view_map_y = find_parameters_to_make_top_view(imgBack_unwarped0)
 
        imgBack_topview = make_top_view(imgBack_unwarped0, shrinking_parameter=self.shrinking_parameter, crop_top=self.crop_top, crop_bottom=self.crop_bottom, map_x=self.top_view_map_x, map_y=self.top_view_map_y)
        imgLeft_topview = make_top_view(imgLeft_unwarped0, shrinking_parameter=self.shrinking_parameter, crop_top=self.crop_top, crop_bottom=self.crop_bottom, map_x=self.top_view_map_x, map_y=self.top_view_map_y)
        imgFront_topview = make_top_view(imgFront_unwarped0, shrinking_parameter=self.shrinking_parameter, crop_top=self.crop_top, crop_bottom=self.crop_bottom, map_x=self.top_view_map_x, map_y=self.top_view_map_y)
        imgRight_topview = make_top_view(imgRight_unwarped0, shrinking_parameter=self.shrinking_parameter, crop_top=self.crop_top, crop_bottom=self.crop_bottom, map_x=self.top_view_map_x, map_y=self.top_view_map_y)
        # img4 = numpy.concatenate((imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview), axis=1)
        # if SHOW_IMAGES is True:
        #     cv2.imshow("Unwarped by undistort in line", cv2.resize(img3, (0, 0), None, 0.5, 0.5))
        #     cv2.imshow("Top view", cv2.resize(img4, (0, 0), None, 0.5, 0.5))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #     sys.exit()
        

        if USE_PREDEFINED_COMBINE_TOP_VIEW_PARAMETERS is True:
            # 1 version
            self.Back_position = [0, 8, 0, 0, 0, 0, 1.38, 0.56]
            self.Left_position = [0, 0, 0, 0, 0, 0, 0.54, 1.56]
            self.Front_position = [0, -8, 0, 0, 0, 0, 1.38, 0.56]
            self.Right_position = [0, 0, 0, 0, 0, 0, 0.54, 1.56]
        
            # 2 version
            # self.Back_position = [0, 0, 0, 0, 0, 0, 1.52, 0.62]
            # self.Left_position = [0, -5, 0, 0, 0, 0, 0.6, 1.54]
            # self.Front_position = [0, 0, 0, 0, 0, 0, 1.52, 0.62]
            # self.Right_position = [0, -5, 0, 0, 0, 0, 0.6, 1.54]
            self.mask_Back, self.mask_Left, self.mask_Front, self.mask_Right = combine_top_view(imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview, self.Back_position, self.Left_position, self.Front_position, self.Right_position, return_masks=True, first_run=False)
        else:
            vertical_offset_for_parallel = 320
            horizontal_offset_for_parallel = 160
            vertical_offset_for_perpendicular = 150
            horizontal_offset_for_perpendicular = 145
            
            vertical_scale_for_parallel = 76
            horizontal_scale_for_parallel = 31
            vertical_scale_for_perpendicular = 30
            horizontal_scale_for_perpendicular = 77
            # vertical_offset_for_parallel = 320
            # horizontal_offset_for_parallel = 160
            # vertical_offset_for_perpendicular = 148
            # horizontal_offset_for_perpendicular = 145
            
            # vertical_scale_for_parallel = 76
            # horizontal_scale_for_parallel = 35
            # vertical_scale_for_perpendicular = 29
            # horizontal_scale_for_perpendicular = 77
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
    
            self.Back_position, self.Left_position, self.Front_position, self.Right_position, self.mask_Back, self.mask_Left, self.mask_Front, self.mask_Right = find_parameters_for_combine_top_view(imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview, self.Back_position, self.Left_position, self.Front_position, self.Right_position)

        
    def calibrate_equirectangular_projection(self):
        """
        Calibrate equirectangular
        """
        if self.frame_counter == 0:
            self.read_frame()
            
            
        # unwarp images using projection
        self.W_remap = 720
        self.H = 640
        self.FOV = 180
        self.equirectangular_xmap, self.equirectangular_ymap = buildmap(Ws=self.W_remap, Hs=self.H, Wd=640, Hd=640, fov=self.FOV)
        imgBack_unwarped = cv2.remap(self.imgBack, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)
        imgLeft_unwarped = cv2.remap(self.imgLeft, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)
        imgFront_unwarped = cv2.remap(self.imgFront, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)
        imgRight_unwarped = cv2.remap(self.imgRight, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)
    
        # img4 = numpy.concatenate((imgBack_unwarped, imgLeft_unwarped, imgFront_unwarped, imgRight_unwarped), axis=1)
        
        # if SHOW_IMAGES is True:
        #     cv2.imshow("Top View image", combined_top_view)cv2.imshow("Unwarped in line", cv2.resize(img4, (0, 0), None, 0.5, 0.5))
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #     sys.exit()
        
                
        if USE_EQUIRECTANGULAR_METHOD is True:
            # Use starting parameters for equirectangular
            self.offsetBackLeft1 = 167
            self.offsetBackLeft2 = 167
            self.offsetLeftFront1 = 167
            self.offsetLeftFront2 = 167
            self.offsetFrontRight1 = 167
            self.offsetFrontRight2 = 167
            self.offsetRightBack1 = 167
            self.offsetRightBack2 = 167
            
            if USE_PREDEFINED_EQURECTANGULAR_PARAMETERS is False:
                # find parameters
                self.offsetBackLeft1, self.offsetBackLeft2 = find_parameters_for_two_image_stack(imgBack_unwarped[:,int(self.W_remap/2):,:], imgLeft_unwarped[:,:int(self.W_remap/2),:], self.offsetBackLeft1, self.offsetBackLeft2)
                self.offsetLeftFront1, self.offsetLeftFront2 = find_parameters_for_two_image_stack(imgLeft_unwarped[:,int(self.W_remap/2):,:], imgFront_unwarped[:,:int(self.W_remap/2),:], self.offsetLeftFront1, self.offsetLeftFront2)
                self.offsetFrontRight1, self.offsetFrontRight2 = find_parameters_for_two_image_stack(imgFront_unwarped[:,int(self.W_remap/2):,:], imgRight_unwarped[:,:int(self.W_remap/2),:], self.offsetFrontRight1, self.offsetFrontRight2)
                self.offsetRightBack1, self.offsetRightBack2 = find_parameters_for_two_image_stack(imgRight_unwarped[:,int(self.W_remap/2):,:], imgBack_unwarped[:,:int(self.W_remap/2),:], self.offsetRightBack1, self.offsetRightBack2)
            
            # concatenate images - stack_two_images_with_offsets - is only for two images, here will be 4, using it 3 times is stupid
            #   then make stake 4 images? more code, but better readability
            # WRITE SIMILAR USING ORB?
            self.equirectangular_image = numpy.concatenate((
                imgRight_unwarped[:,int(self.W_remap/2):-self.offsetRightBack1,:],
                imgBack_unwarped[:,self.offsetRightBack2:-self.offsetBackLeft1,:],
                imgLeft_unwarped[:,self.offsetBackLeft2:-self.offsetLeftFront1,:],
                imgFront_unwarped[:,self.offsetLeftFront2:-self.offsetFrontRight1,:],
                imgRight_unwarped[:,self.offsetFrontRight2:int(self.W_remap/2),:]
                ), axis=1)
        
        elif USE_ORB_IN_EQUIRECTANGULAR_METHOD is True:
        
            #stitch unwarped images
            # self.vertical_stitching_offset = 40
            self.vertical_stitching_offset = 30
            self.horizontal_stitching_offset = 80
            # Zrobic to tak, ze funkcja dostaje mniejsze ROI do poszukiwania cech wspolnych, a pozniej przesunac punkty, bedzie szbysze i dokladniejsze
            #opcja z doklejaniem
            # stitched_BL, self.parameters_BL = stitch_two_images_using_ORB(imgBack_unwarped[self.horizontal_stitching_offset:H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :],imgLeft_unwarped[self.horizontal_stitching_offset:self.H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :])
            # stitched_BLF, self.parameters_BLF = stitch_two_images_using_ORB(stitched_BL,imgFront_unwarped[self.horizontal_stitching_offset:self.H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :])
            # stitched_BLFR, self.parameters_BLFR = stitch_two_images_using_ORB(stitched_BLF,imgRight_unwarped[self.horizontal_stitching_offset:self.H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :])
            # print("BLFR\n")
            # print(stitched_BLFR.shape)
            #opcja rownolegla
            stitched_BL, self.parameters_BL = stitch_two_images_using_ORB(imgBack_unwarped[self.horizontal_stitching_offset:self.H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :],imgLeft_unwarped[self.horizontal_stitching_offset:self.H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :])
            stitched_FR, self.parameters_FR = stitch_two_images_using_ORB(imgRight_unwarped[self.horizontal_stitching_offset:self.H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :],imgFront_unwarped[self.horizontal_stitching_offset:self.H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :])
            stitched_BLRF, self.parameters_BLRF = stitch_two_images_using_ORB(stitched_BL,stitched_FR)
            # print("BLRF\n")
            # print(stitched_BLRF.shape)
            
            # crop image - czy to ma sens? - raczej nie, zobaczymy jak to będzie dzialac dalej, teraz gupie
            # moze wypadaloby zrobic projekcje z "equrectangular"/"obround" do "rectangular" przedtym
            height_in_stiched = numpy.nonzero(stitched_BLRF[:,int(stitched_BLRF.shape[1]/7),:][:,1])[0]
            width_in_stiched = numpy.nonzero(stitched_BLRF[int(stitched_BLRF.shape[0]/2),:,:][:,1])[0]
            croped_stiched_BLRF = stitched_BLRF[min(height_in_stiched):max(height_in_stiched), min(width_in_stiched):max(width_in_stiched),:]
            self.equirectangular_image = croped_stiched_BLRF
        
            # Co tu było właciwie zamierzone?
            #find parameters
            self.offsetBackLeft1, self.offsetBackLeft2 = find_parameters_for_two_image_stack(self.imgBack,self.imgLeft)
            stacked_back_left = stack_two_images_with_offsets(self.imgBack,self.imgLeft,self.offsetBackLeft1,self.offsetBackLeft2)
            
            self.offsetBackLeftFront1, self.offsetBackLeftFront2 = find_parameters_for_two_image_stack(stacked_back_left,self.imgFront)
            stacked_back_left_front = stack_two_images_with_offsets(stacked_back_left,self.imgFront,self.offsetBackLeftFront1,self.offsetBackLeftFront2)
            
            self.offsetBackLeftFrontRight1, self.offsetBackLeftFrontRight2 = find_parameters_for_two_image_stack(stacked_back_left_front,self.imgRight)
            stacked_back_left_front_right = stack_two_images_with_offsets(stacked_back_left_front,self.imgRight,self.offsetBackLeftFrontRight1,self.offsetBackLeftFrontRight2)
            
            # if SHOW_IMAGES is True:
            #     cv2.imshow("stacked_back_left_front_right", stacked_back_left_front_right)    
            #     cv2.waitKey(0)
            #     #end find parameters
            
        else:
            raise SyntaxError("USE_EQUIRECTANGULAR_METHOD or USE_ORB_IN_EQUIRECTANGULAR_METHOD must be True")
        
        
    def read_frame(self):
        successBack, self.imgBack = self.capBack.read()
        successLeft, self.imgLeft = self.capLeft.read()
        successFront, self.imgFront = self.capFront.read()
        successRight, self.imgRight = self.capRight.read()
        
        if successBack is True and successLeft is True and successFront is True and successRight is True:
            self.frame_read_successfully = True
            self.frame_counter += 1
        else:
            # DO ZMIANY
            self.frame_read_successfully = False
            
        if CAMERA_READ_FROM_FILE is True:
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
        # combined_top_view = combine_top_view(imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview, self.Back_position, self.Left_position, self.Front_position, self.Right_position, self.mask_Back, self.mask_Left, self.mask_Front, self.mask_Right)
        self.top_view_image = combine_top_view(imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview, self.Back_position, self.Left_position, self.Front_position, self.Right_position, self.mask_Back, self.mask_Left, self.mask_Front, self.mask_Right)
        # time3 = time.time()
        
        # height_in_stiched = numpy.nonzero(combined_top_view[:,int(combined_top_view.shape[1]/2),:][:,1])[0]
        # width_in_stiched = numpy.nonzero(combined_top_view[int(combined_top_view.shape[0]/2),:,:][:,1])[0]
        # self.top_view_image = combined_top_view[min(height_in_stiched):max(height_in_stiched), min(width_in_stiched):max(width_in_stiched),:]
        # time4 = time.time()
            
        # print("Time in top_view {:.4f} make {:.4f} combine {:.4f} total {:.4f} seconds".format(time1-time0, time2-time1, time3-time2, time3-time0))
                
        
        # cv2.imwrite("prezentacja/imgBack_unwarped0.jpg", imgBack_unwarped0)
        # cv2.imwrite("prezentacja/imgBack_topview.jpg", imgBack_topview)
        # cv2.imwrite("prezentacja/top_view_image.jpg", self.top_view_image)
        
        # if SHOW_IMAGES is True:
        #     cv2.imshow("self.top_view_image", self.top_view_image)
        #     # cv2.imshow("combined_top_view", combined_top_view)
        #     cv2.waitKey(0)
        #     cv2.destroyWindow("self.top_view_image")
        #     # cv2.destroyAllWindows()
        #     # sys.exit()
        
 
    def equirectangular_projection(self):
        imgBack_unwarped = cv2.remap(self.imgBack, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)
        imgLeft_unwarped = cv2.remap(self.imgLeft, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)
        imgFront_unwarped = cv2.remap(self.imgFront, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)
        imgRight_unwarped = cv2.remap(self.imgRight, self.equirectangular_xmap, self.equirectangular_ymap, cv2.INTER_LINEAR, cv2.CV_32FC1)
        
        if USE_EQUIRECTANGULAR_METHOD is True:
            self.equirectangular_image = numpy.concatenate((
                imgRight_unwarped[:,int(self.W_remap/2):-self.offsetRightBack1,:],
                imgBack_unwarped[:,self.offsetRightBack2:-self.offsetBackLeft1,:],
                imgLeft_unwarped[:,self.offsetBackLeft2:-self.offsetLeftFront1,:],
                imgFront_unwarped[:,self.offsetLeftFront2:-self.offsetFrontRight1,:],
                imgRight_unwarped[:,self.offsetFrontRight2:int(self.W_remap/2),:]
                ), axis=1)
            
        elif USE_ORB_IN_EQUIRECTANGULAR_METHOD is True: # o co tu właciwie chodziło
            # DO ZMIANY
            # Zrobic to tak, ze funkcja dostaje mniejsze ROI do poszukiwania cech wspolnych, a pozniej przesunac punkty, bedzie szbysze i dokladniejsze
            #opcja z doklejaniem
            # stitched_BL, self.parameters_BL = stitch_two_images_using_ORB(imgBack_unwarped[self.horizontal_stitching_offset:H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :],imgLeft_unwarped[self.horizontal_stitching_offset:self.H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :])
            # stitched_BLF, self.parameters_BLF = stitch_two_images_using_ORB(stitched_BL,imgFront_unwarped[self.horizontal_stitching_offset:self.H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :])
            # stitched_BLFR, self.parameters_BLFR = stitch_two_images_using_ORB(stitched_BLF,imgRight_unwarped[self.horizontal_stitching_offset:self.H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :])
            # print("BLFR\n")
            # print(stitched_BLFR.shape)
            #opcja rownolegla
            stitched_BL, self.parameters_BL = stitch_two_images_using_ORB(imgBack_unwarped[self.horizontal_stitching_offset:self.H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :],imgLeft_unwarped[self.horizontal_stitching_offset:self.H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :])
            stitched_FR, self.parameters_FR = stitch_two_images_using_ORB(imgRight_unwarped[self.horizontal_stitching_offset:self.H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :],imgFront_unwarped[self.horizontal_stitching_offset:self.H-self.horizontal_stitching_offset, self.vertical_stitching_offset:self.W_remap-self.vertical_stitching_offset, :])
            stitched_BLRF, self.parameters_BLRF = stitch_two_images_using_ORB(stitched_BL,stitched_FR)
            # print("BLRF\n")
            # print(stitched_BLRF.shape)
            
            # crop image - czy to ma sens? - raczej nie, zobaczymy jak to będzie dzialac dalej, teraz gupie
            # moze wypadaloby zrobic projekcje z "equrectangular"/"obround" do "rectangular" przedtym
            height_in_stiched = numpy.nonzero(stitched_BLRF[:,int(stitched_BLRF.shape[1]/7),:][:,1])[0]
            width_in_stiched = numpy.nonzero(stitched_BLRF[int(stitched_BLRF.shape[0]/2),:,:][:,1])[0]
            croped_stiched_BLRF = stitched_BLRF[min(height_in_stiched):max(height_in_stiched), min(width_in_stiched):max(width_in_stiched),:]
            self.equirectangular_image = croped_stiched_BLRF
            
        
        # if SHOW_IMAGES is True:
        #     cv2.imshow("self.equirectangular_image", self.equirectangular_image)
        #     cv2.imwrite("prezentacja/imgBack_unwarped.jpg", imgBack_unwarped)
        #     cv2.imwrite("prezentacja/imgBack.jpg", self.imgBack)
        #     cv2.imwrite("prezentacja/equirectangular_image.jpg", self.equirectangular_image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        #     sys.exit()
            
            
    def run(self, dont_stop = True):
        if self.frame_counter == 0:
            self.read_frame()
        self.FPS = []
        while dont_stop:
            if cv2.waitKey(1) & 0xFF == ord('q') or self.frame_read_successfully is False or dont_stop is False:
                break
            else:
                time0 = time.time()
                
                if MAKE_TOP_VIEW is True:
                    self.top_view()
                time1 = time.time()

                if MAKE_EQUIRECTANGULAR_PROJECTION is True:
                    self.equirectangular_projection()
                time2 = time.time()
                
                self.read_frame()
                time3 = time.time()
                
                if SHOW_IMAGES is True:
                    if MAKE_TOP_VIEW is True:
                        cv2.imshow("self.top_view_image", self.top_view_image)
                    if MAKE_EQUIRECTANGULAR_PROJECTION is True:
                        cv2.imshow("self.equirectangular_image", self.equirectangular_image)
                        
                    # time.sleep(0.2)

                # tymczasowy warunek stopu pętli
                else:
                    if self.frame_counter == 10:
                        dont_stop = False
                                            
                time4 = time.time()
                    
                # print("Time in top_view {:.4f} equi {:.4f} read {:.4f} show {:.4f} total {:.4f} seconds".format(time1-time0, time2-time1, time3-time2, time4-time3, time4-time0))
                self.FPS.append(time4-time0)
                
                
    # def get_top_view_image(self):
    #     return self.top_view_image
    
    # def get_equirectangular_image(self):
    #     return self.equirectangular_image
    
    def __del__(self):
        cv2.destroyAllWindows()
        self.capBack.release()
        self.capLeft.release()
        self.capFront.release()
        self.capRight.release()
        print("Average FPS {}".format(1/(sum(self.FPS)/len(self.FPS))))


"""
Start here
"""

if __name__ == '__main__':
    # Declare parameters of program
    USE_PREDEFINED_CAMERA_PARAMETERS = True
    ONLY_VALID_IMAGES_FOR_CAMERA_CALIBRATION = False
    CAMERA_READ_FROM_FILE = True # polaczyc z __init__
    
    MAKE_TOP_VIEW = True
    USE_PREDEFINED_TOP_VIEW_PARAMETERS = True
    USE_PREDEFINED_COMBINE_TOP_VIEW_PARAMETERS = True # False - ale ciagle sa wstepne
    
    MAKE_EQUIRECTANGULAR_PROJECTION = True
    
    USE_EQUIRECTANGULAR_METHOD = True # False doesn't work for now
    USE_PREDEFINED_EQURECTANGULAR_PARAMETERS = True # False - ale ciagle sa wstepne
    USE_ORB_IN_EQUIRECTANGULAR_METHOD = not USE_EQUIRECTANGULAR_METHOD
    
    SHOW_IMAGES = True
    
    assert MAKE_TOP_VIEW is True or MAKE_EQUIRECTANGULAR_PROJECTION is True
    
    if ONLY_VALID_IMAGES_FOR_CAMERA_CALIBRATION is True:
        only_valid_images_for_calibration = []
        only_valid_images_for_calibration.append("dataset5/0009.jpg")
        only_valid_images_for_calibration.append("dataset5/0010.jpg")
        only_valid_images_for_calibration.append("dataset5/0023.jpg")
        only_valid_images_for_calibration.append("dataset5/0032.jpg")
        only_valid_images_for_calibration.append("dataset5/0058.jpg")
        only_valid_images_for_calibration.append("dataset5/0067.jpg")
    
    main = Main()
    # main.top_view()
    # main.equirectangular_projection()
    main.run(dont_stop = True)
    del main
    # try:
    #     main()
    # except Exception:
    #     print("type: \t\t", sys.exc_info()[0].__name__, 
    #           "\nfilename: \t", os.path.split(sys.exc_info()[2].tb_frame.f_code.co_filename)[1], 
    #           "\nlineo: \t\t", sys.exc_info()[2].tb_lineno,
    #           #"\nname: \t", sys.exc_info()[2].tb_frame.f_code.co_name,
    #           "\nmessage: \t", sys.exc_info()[1])

# #test laczenia zdjec
# testowe0 = cv2.imread("zdjTestPanoramy2.png")
# testowe1 = cv2.imread("zdjTestPanoramy3.png")

# stitched_using_orb, parameters = stitch_two_images_using_ORB(cv2.resize(testowe0, (0, 0), None, 0.5, 0.5), cv2.resize(testowe1, (0, 0), None, 0.5, 0.5))
# stitched_using_orb_and_parameters, parameters2 = stitch_two_images_using_ORB(cv2.resize(testowe0, (0, 0), None, 0.5, 0.5), cv2.resize(testowe1, (0, 0), None, 0.5, 0.5), parameters=parameters)
# assert parameters == parameters2
# cv2.imshow("Result", stitched_using_orb)
# cv2.imshow("Result_with_parameters", stitched_using_orb_and_parameters)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #end test laczenia zdjec

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
# Back_position, Left_position, Front_position, Right_position = find_parameters_for_combine_top_view(imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview, Back_position, Left_position, Front_position, Right_position)
# combined_top_view = combine_top_view(imgBack_topview, imgLeft_topview, imgFront_topview, imgRight_topview, Back_position, Left_position, Front_position, Right_position)

# cv2.imshow("Result of top view test", cv2.resize(combined_top_view, (0, 0), None, 0.8, 0.8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()