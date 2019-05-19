import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from simple_board2d import *
from padPhoto import *
from chessboardPoints import *
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
import math
import random
import copy

import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')



#Put everything together
def getHomographyMat(originalBoard, rotatedBoard):
    h_matrix = create_h_matrix(originalBoard, rotatedBoard)
    solvedMatrix = solve(h_matrix)
    return solvedMatrix

#CREATE IMAGE MATRIX
def makeImgMat(img):
    # get dimensions of image
    dimensions = img.shape
    # height, width, number of channels in image
    height = dimensions[0]
    width = dimensions[1]
    channels = dimensions[2]

    print('Image Dimension    : ',dimensions)
    print('Image Height       : ',height)
    print('Image Width        : ',width)
    print('Number of Channels : ',channels) 

    num_pnts = height * width
    xMat = np.full(num_pnts, 0.0, dtype = float)
    yMat = np.full(num_pnts, 0.0, dtype = float)
    wMat = np.full(num_pnts, 1.0, dtype = float)
    index = 0
    for r in range(0, height):
        for c in range (0, width):
            xMat[index] = c
            yMat[index] = r
            index = index + 1
    xMat = xMat.reshape(height, width)
    yMat = yMat.reshape(height, width)
    wMat = wMat.reshape(height, width)
    return xMat, yMat, wMat, height, width
    
# Returns 3 matrices, one for x, y and w.  Contains points altered
# by homography matrix.  Each row and column corresponds to a pixel.
def makeTransformedImage(hMat, x, y, w, height, width):
    for r in range(0, height):
        for c in range(0, width):
            temp = np.array([x[r][c], y[r][c], w[r][c]])
            temp = np.reshape(temp, (3,1))
            temp = np.dot(hMat, temp)
            #temp = temp.reshape((temp.shape[1], 1))
            x[r][c] = temp[0,0]
            y[r][c] = temp[1,0]
            w[r][c] = temp[2,0]
    return x, y, w
    
def copy_paste(chessImg, img3, corners):
    dimensions = chessImg.shape
    # height, width, number of channels in image
    height = dimensions[0]
    width = dimensions[1]
    chessImg = Image.fromarray(chessImg)

    #print(corners)
    #print(chessImg.size)
    #print(img3.size)
    x1 = int(round(corners[0,0]))
    y1 = int(round(corners[1,0]))
    x2 = int(round(corners[0,-7]))
    y2 = int(round(corners[1,-7]))


    gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    #cv2.imshow('thresh', thresh)
    #cv2.waitKey(0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    x,y,w,h = cv2.boundingRect(cnt)
    img2 = img3[y:y+h,x:x+w]


    img1 = np.array(chessImg)
    
    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[y2:rows+y2, x2:cols+x2]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    #print(roi)
    cv2.imshow('roi',roi)
    cv2.waitKey(0)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[y2:rows+y2, x2:cols+x2 ] = dst

    return img1


def indHomTrans(imgFLIP, board2D, board2DTrans, w1, h1, num_pnts, img1):
    img1 = np.array(img1)
    dimensions = img1.shape
    # height, width, number of channels in image
    height = dimensions[0]
    width = dimensions[1]

    cv2.imshow('img1',img1)
    #numP = 0
    index = 0

    for numP in range(0, num_pnts-1):
        for i in range(0, num_pnts-1):
            nx1 = int(board2D[0,i+(numP*num_pnts)])
            nx2 = int(board2D[0,i+(numP*num_pnts)+1])
            ny1 = int(board2D[1,i+(numP*num_pnts)])
            ny2 = int(board2D[1,i+(numP*num_pnts)+num_pnts])

            sub_board2D = np.reshape(board2D[:, numP*num_pnts+i], (3,1))
            sub_board2D = np.concatenate((sub_board2D, np.reshape(board2D[:, numP*num_pnts+(i+1)], (3,1))), axis = 1)
            sub_board2D = np.concatenate((sub_board2D, np.reshape(board2D[:, (numP+1)*num_pnts+(i)], (3,1))), axis = 1)
            sub_board2D = np.concatenate((sub_board2D, np.reshape(board2D[:, (numP+1)*num_pnts+(i+1)], (3,1))), axis = 1)
            sub_board2DTrans = np.reshape(board2DTrans[:, numP*num_pnts+i],(3,1))
            sub_board2DTrans = np.concatenate((sub_board2DTrans, np.reshape(board2DTrans[:, numP*num_pnts+(i+1)], (3,1))), axis = 1)
            sub_board2DTrans = np.concatenate((sub_board2DTrans, np.reshape(board2DTrans[:, (numP+1)*num_pnts+(i)], (3,1))), axis = 1)
            sub_board2DTrans = np.concatenate((sub_board2DTrans, np.reshape(board2DTrans[:, (numP+1)*num_pnts+(i+1)], (3,1))), axis = 1)
            
            im = imgFLIP[ny2:ny1,nx1:nx2]

            hom = getHomographyMat(sub_board2D, sub_board2DTrans)
            img3 = cv2.warpPerspective(im, hom, (w1,h1))
            
            index= index+1

            x1 = int(round(board2DTrans[0,i+(numP*num_pnts)]))
            y1 = int(round(board2DTrans[1,i+(numP*num_pnts)]))
            x2 = int(round(board2DTrans[0,i+(numP*num_pnts)+1]))
            y2 = int(round(board2DTrans[1,i+(numP*num_pnts)+num_pnts]))
            #print(y1)
            gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
            _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
            
            contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = contours[0]
                
                x,y,w,h = cv2.boundingRect(cnt)
                img2 = img3[y:y+h,x:x+w]
                img2 = cv2.resize(img2, None, fx=1.1, fy=1.1, interpolation = cv2.INTER_LINEAR)    
                # I want to put logo on top-left corner, So I create a ROI
                rows,cols,channels = img2.shape
                roi = img1[y2:rows+y2, x1:cols+x1]
                # Now create a mask of logo and create its inverse mask also
                img2gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
                ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                #print(roi)
                
                # Now black-out the area of logo in ROI
                img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
                # Take only region of logo from logo image.
                img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
                # Put logo in ROI and modify the main image
                dst = cv2.add(img1_bg,img2_fg)
                img1[y2:rows+y2, x1:cols+x1 ] = dst
                cv2.imshow('img1',img1)
                #cv2.waitKey(0)
            
            else:
                print(index)

    return img1
            

# Make image and image matrices
img = mpimg.imread('logo.jpg')
img = img.copy()
img[np.where((img==[0,0,0]).all(axis=2))]=[2,2,2]
cam_img = mpimg.imread('board8.jpg')
img = makeSquare(img)

imgFLIP = img[::-1,:,:]

# Each is a matrix representing the x, y, or w value in the matrix
xMat, yMat, wMat, height, width = makeImgMat(cam_img)

#Create original board
num_pnts = 7

board2D = create_board2D(num_pnts, img)

chessImg, board2DTrans = getChessboardCorners(cam_img)
carB2DT = hom_cart_trans(board2DTrans) #delete
plt.subplot(2, 2, 1)
plt.plot(carB2DT[0,:], carB2DT[1,:], 'bo')
#plt.plot(carB2DT[0,0], 'ro')
plt.grid(True)



ax1 = plt.subplot(2, 2, 2)
ax1.imshow(imgFLIP, origin = 'lower')


ax2 = plt.subplot(2, 2, 3)
#img2 = cv2.remap(img, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_LINEAR) #how to fix lmao
ax2.imshow(chessImg, origin = 'lower')

img4 = indHomTrans(imgFLIP, board2D, board2DTrans, width,height,num_pnts,chessImg)
#img4 = cv2.warpPerspective(imgFLIP[100:100+int(width/(num_pnts-2)),100:100+int(height/(num_pnts-2))], homs[2], (width,height))
ax4 = plt.subplot(2, 2, 4)
ax4.imshow(img4, origin = 'lower')
plt.show()

# cv2.imshow('img3',img3)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()
#cv2.imshow('image',img2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


print('end')


