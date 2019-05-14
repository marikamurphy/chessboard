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

img = mpimg.imread('logo.jpg')
imgplot = plt.imshow(img)
plt.show()

# Create the boards
num_pnts = 5
board3D = create_board3D(num_pnts)
board2D = hom_3Dto2D(board3D)

# Create a random transformation matrix
rot_mat = random_trans_generator() #arbitrary rotation
rigid_trans_mat = rigid_trans(rot_mat, [[0],[0], [0]])
trans_mat = transform_matrix(board3D, rigid_trans_mat)

board2DTrans = hom_3Dto2D(trans_mat)
plot_board_2d(hom_cart_trans(board2DTrans), 'bo')

hMat = create_h_matrix(board2D, board2DTrans)

solved = solve(hMat)


#print(temp)
#print("hmat")
#print(hMat)
#solved = solve(hMat)
#print("solved?")
b1 = board2D[:2].transpose()
b2 = board2DTrans[:2].transpose()
temp = cv2.findHomography(b1, b2)
print(temp[0])

product2 = np.dot(temp[0],board2D)
product = np.dot(solved,board2D)

plot_board_2d(product, 'go')
plot_board_2d(product2, 'r.')


#print(temp[0])
#print(solved)
plt.show()
print('end')