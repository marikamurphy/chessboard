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
    iMat = np.full(num_pnts, 0, dtype = int)


img = mpimg.imread('logo.jpg')


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
hMat = findHomography(board2D, board2DTrans)
#plot_board_2d(product, 'r*')
imgplot = plt.imshow(img)

plt.show()
print('end')

makeImgMat(img)

