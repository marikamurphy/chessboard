import math
import glob
import webbrowser
import argparse
import sys
import numpy as np
import sys
import cv2

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.




# orb = cv2.ORB_create()

# #images = glob.glob('board1.jpg')


#clahe = cv2.createCLAHE();
#clahe.apply(img, img)
#corners

# kp_grayframe = orb.detect(gray, None)

# # Find the chess board corners


# img2 = cv2.drawKeypoints(img,kp_grayframe, img)

# cv2.imshow('img',img2)

# If found, add object points, image points (after refining them)
#

    # Draw and display the corners


#corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#imgpoints.append(corners2)

#(thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
def getChessboardCorners(img):
    DEFAULT_WIDTH = 500
    #img = cv2.imread('board6.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh = 127
    im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("bw", im_bw)

    height, width = gray.shape[:2]

    scaleFactor0 = DEFAULT_WIDTH/width

    gray1 = cv2.resize(im_bw, None, fx=scaleFactor0, fy=scaleFactor0, interpolation = cv2.INTER_LINEAR)
    img1 = cv2.resize(img, None, fx=scaleFactor0, fy=scaleFactor0, interpolation = cv2.INTER_LINEAR)
    dimensions = (7,7)
    ret, corners = cv2.findChessboardCorners(gray1, dimensions , None)
    cv2.drawChessboardCorners(img1, dimensions, corners , ret)
    corners = getHom2DMat(corners, img1)
    return img1[::-1], corners

def getHom2DMat(corners, img):
    height = img.shape[0]
    num_pnts = corners.shape[0]
    board_hom = np.zeros(num_pnts*3)
    board_hom = np.reshape(board_hom, (3, num_pnts))

    for i in range(0, num_pnts):
        board_hom[0,i] = corners[i, 0, 0] #x
        board_hom[1,i] = height-corners[i, 0, 1] #y
        board_hom[2,i] = 1 #w


    return board_hom


if __name__ == '__main__':
    img1, corners = getChessboardCorners()
    print(corners)
    print("-------------------------------------------------------------")
    #corners = getHom2DMat(corners)
    #print(corners)
    cv2.imshow('img1',img1)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

print('end')
# Find the chess board corners


# ret, corners = cv2.findChessboardCorners(gray, (6,6) ,None)

# # If found, add object points, image points (after refining them)
# if ret == True:
#     #objpoints.append(objp)

#     corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#     #imgpoints.append(corners2)

#     # Draw and display the corners
#     img2 = cv2.drawChessboardCorners(img, (6,6), corners2,ret)
#     cv2.imshow('img',img2)
    

# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#     # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)

#         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         imgpoints.append(corners2)

#         # Draw and display the corners
#         img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
#         cv2.imshow('img',img)


