#simple plot of checkerboard points and entire board shifted +1 in y-direction
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
import math
import random

import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


def create_board3D(num_pnts):
	board = np.full(num_pnts, 0, dtype = int) #(this is x) 0, 0, 0...1, 1, ..
	y = np.arange(0, num_pnts) # 0, 1 ... num_pnts-1
	z = np.full(num_pnts*num_pnts, 0, dtype = int) #all 0s
	w = np.full(num_pnts*num_pnts, 1, dtype = int) # all 1s

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
	

def hom_cart_trans(board):
	board_cart = np.array(board) #otherwise we modify board passed into method
	board_cart = board_cart[:-1,:] / board[-1,:]
	return board_cart

def hom_3Dto2D(board3D):
	board2D = np.array(board3D)
	board2D = board2D / board2D[-1,:]
	return np.concatenate((board2D[:2,:], [board2D[-1]]), axis = 0)

def rot_mat(rot1, rot2, rot3):
	rot_vec = np.array([rot1,rot2,rot3])
	rot_mat = np.zeros((3,3))
	cv2.Rodrigues(rot_vec,rot_mat)
	return rot_mat #3x3

def rigid_trans(rot_mat, trans_mat):
	trans_mat = np.array(trans_mat)
	RT_mat = np.concatenate((rot_mat, trans_mat), axis = 1)
	
	RT_mat = np.concatenate((RT_mat, np.array([[0, 0, 0, 1]]) ), axis = 0)
	# this is our {R  T}
	#              {0001}
	return RT_mat # 4x4

def transform_matrix(board, rigid_trans_mat):
	return np.dot(rigid_trans_mat, board)	

def plot_board_2d(board, marker):
	plt.plot(board[0,:], board[1,:], marker)
	plt.grid(True)

def random_trans_generator():
	rand1 = random.random()*math.pi
	rand2 = random.random()*math.pi
	rand3 =	random.random()*math.pi
	
	return rot_mat(rand1,rand2,rand3)		

if __name__ == '__main__':

	num_pnts = 5
 
	board3D = create_board3D(num_pnts)
	#board_3Dcart = hom_cart_trans(board3D)
	#print("board3D")
	#print(board3D)
	#print("board_3Dcart")
	#print(board_3Dcart)
	plot_board_2d(hom_cart_trans(board3D), 'ro')
	board2D = create_board2D(num_pnts)
	board_2Dcart = hom_cart_trans(board2D)
	#print("board2D")
	#print(board2D)
	#print("board_2Dcart")
	#print(board_2Dcart)

	print("rigid transformation")
	rot_mat = random_trans_generator() #arbitrary rotation
	rigid_trans_mat = rigid_trans(rot_mat, [[0],[0], [0]])
	trans_mat = transform_matrix(board3D, rigid_trans_mat)
	plot_board_2d(hom_cart_trans(trans_mat), 'b*')
	
	print(hom_cart_trans(trans_mat))

	board2D = hom_3Dto2D(trans_mat)
	#print("trans_mat")
	#print(trans_mat)
	#print("board2D")
	#print(board2D)
	plt.show()

