
#simple plot of checkerboard points and entire board shifted +1 in y-direction
import numpy as np
import matplotlib.pyplot as plt

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

#Creates an NxN 3D board with rows x, y, z, w
def create_board3D(num_pnts):
    # Returns a new array with fill value, in this case 0
	board = np.full(num_pnts, 0, dtype = int) #(this is x) 0, 0, 0...1, 1, ..
	y = np.arange(0, num_pnts) # 0, 1 ... num_pnts-1
	z = np.full(num_pnts*num_pnts, 0, dtype = int) #all 0s
	w = np.full(num_pnts*num_pnts, 1, dtype = int) # all 1s

    #concatenate the arrays together
	for num in range(1, num_pnts):
		y = np.concatenate((y,  np.arange(0, num_pnts)), axis = 0)
		board = np.concatenate((board,  np.full(num_pnts, num, dtype =int)), axis = 0)

	board = np.concatenate((board, y), axis = 0)
	board = np.concatenate((board, z), axis = 0)
	board = np.concatenate((board, w), axis = 0)
	board = np.reshape(board, (4, num_pnts * num_pnts)) #its one long array -> need to reshape
	#[x,y,z,w] becomes ...
	#[[x],
	# [y],
	# [z],
	# [w]]

	return board

#Creates an NxN 2D board with rows x, y, w
def create_board2D(num_pnts):
	board = np.full(num_pnts, 0, dtype = int) #(this is x) 0, 0, 0...1, 1, ..
	y = np.arange(0, num_pnts) # 0, 1 ... num_pnts-1
	w = np.full(num_pnts*num_pnts, 1, dtype = int) # all 1s

	for num in range(1, num_pnts):
		y = np.concatenate((y,  np.arange(0, num_pnts)), axis = 0)
		board = np.concatenate((board,  np.full(num_pnts, num, dtype =int)), axis = 0)

	board = np.concatenate((board, y), axis = 0)
	board = np.concatenate((board, w), axis = 0)
	board = np.reshape(board, (3, num_pnts * num_pnts)) #its one long array -> need to reshape
	#[x,y,z,w] becomes ...
	#[[x],
	# [y],
	# [w]]

	return board

#Transforms a homogenous plane to cartesian plane
def hom_cart_trans(board):
	board_cart = np.array(board) #otherwise we modify board passed into method
	board_cart = board_cart[:-1,:] / board[-1,:]
	return board_cart

#Transforms 3D matrix into a 2D Matrix
def hom_3Dto2D(board3D):
	board2D = np.array(board3D)
	board2D = board2D / board2D[-1,:]
	return np.concatenate((board2D[:2,:], [board2D[-1]]), axis = 0)

#Rotates the matrix
def rot_mat(rot1, rot2, rot3):
	rot_vec = np.array([rot1,rot2,rot3])
	rot_mat = np.zeros((3,3))
	cv2.Rodrigues(rot_vec,rot_mat)
	return rot_mat #3x3

#Creates a rigid transformation matrix with rotation and translation
def rigid_trans(rot_mat, trans_mat):
	trans_mat = np.array(trans_mat)
	RT_mat = np.concatenate((rot_mat, trans_mat), axis = 1)

	RT_mat = np.concatenate((RT_mat, np.array([[0, 0, 0, 1]]) ), axis = 0)
	# this is our {R   T}
	#             {000 1}
	return RT_mat # 4x4

#Apply rigid transformation to matrix
def transform_matrix(board, rigid_trans_mat):
	return np.dot(rigid_trans_mat, board)

#Plot the board onto a graphics simulator
def plot_board_2d(board, marker):
	plt.plot(board[0,:], board[1,:], marker)
	plt.grid(True)

#generate random rotations and translations
def random_trans_generator():
	rand1 = random.random()*math.pi
	rand2 = random.random()*math.pi
	rand3 =	random.random()*math.pi

	return rot_mat(rand1,rand2,rand3)

#Building the homography matrix
#takes in the original board and the rotated board in 2D
def create_h_matrix(originalBoard, rotatedBoard):

    #original
    h_list = []
    for c in range(0, len(originalBoard[0])):
        #original matrix
        x = originalBoard[0,c]
        y = originalBoard[1,c]
        w = originalBoard[2,c]
        #rotated matrix
        _x = rotatedBoard[0, c]
        _y = rotatedBoard[1, c]
        _w = rotatedBoard[2, c]

        point_matrix_A = [0, 0, 0, -_w*x, -_w*y, -_w*w, _y*x,_y*y, _y*w]
        point_matrix_B = [-_w*x, -_w*y, -_w*w, 0, 0, 0, _x*x, _x*y, _x*w]
        h_list.append(point_matrix_B)
        h_list.append(point_matrix_A)

    h_matrix = np.matrix(h_list)
    return h_matrix

#Apply DLT to homography and transform original matrix to rotated
def solve(matrix):
    temp = copy.deepcopy(matrix)
    u, s, v = np.linalg.svd(temp)
    h = np.reshape(v[8],(3,3))
    h = (1/h.item(8))*h
    return h

#Put everything together
def findHomography(originalBoard, rotatedBoard):
    h_matrix = create_h_matrix(originalBoard, rotatedBoard)
    solvedMatrix = solve(h_matrix)
    product = np.dot(solvedMatrix, originalBoard)
    return product


if __name__ == '__main__':

    #Create original board
    num_pnts = 5
    board3D = create_board3D(num_pnts)
    board2D = hom_3Dto2D(board3D)

    #Create rotated board
    rot_mat = random_trans_generator() #arbitrary rotation
    rigid_trans_mat = rigid_trans(rot_mat, [[0],[0], [0]])
    trans_mat = transform_matrix(board3D, rigid_trans_mat)
    board2DTrans = hom_3Dto2D(trans_mat)
    plot_board_2d(hom_cart_trans(board2DTrans), 'bo')

    #find Homography and apply to original
    product = findHomography(board2D, board2DTrans)
    plot_board_2d(product, 'r*')

    plt.show()
    print('end')
