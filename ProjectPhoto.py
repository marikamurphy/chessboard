import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from simple_board2d import *

from mpl_toolkits.mplot3d import Axes3D
import math
import random
import copy

import sys
import cv2


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
    xMat = np.full(num_pnts, 0, dtype = int)
    yMat = np.full(num_pnts, 0, dtype = int)
    wMat = np.full(num_pnts, 1, dtype = int)
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
    
def makeTransformedImage(hMat, x, y, w, height, width):
    for r in range(0, height):
        for c in range(0, width):
            temp = [x[r][c], y[r][c], w[r][c]]
            temp = temp @ hMat
            
            # a = temp.split()
            # print('a')
            # print(a[2])
            # print(hMat.shape)
            # print(temp[0][0])
            # x[r][c] = temp[0][0]
            # y[r][c] = temp[0][1]
            # w[r][c] = temp[0][2]
    return x, y, w
    

# Make image and image matrices
img = mpimg.imread('logo.jpg')
# Each is a matrix representing the x, y, or w value in the matrix
xMat, yMat, wMat, height, width = makeImgMat(img)

#Create original board
num_pnts = 5
board3D = create_board3D(num_pnts)
board2D = hom_3Dto2D(board3D)

#Create rotated board
rot_mat = random_trans_generator() #arbitrary rotation
rigid_trans_mat = rigid_trans(rot_mat, [[0],[0], [0]])
trans_mat = transform_matrix(board3D, rigid_trans_mat)
board2DTrans = hom_3Dto2D(trans_mat)
carB2DT =hom_cart_trans(board2DTrans)
plt.plot(carB2DT[0,:], carB2DT[1,:], 'bo')
plt.grid(True)


#find Homography and apply to original
hMat = getHomographyMat(board2D, board2DTrans)
x, y, w = makeTransformedImage(hMat, xMat, yMat, wMat, height, width)
print(x)
print(y)
print(w)
#plot_board_2d(product, 'r*')
imgplot = plt.imshow(img)

plt.show()
print('end')



