#simple plot of checkerboard points and entire board shifted +1 in y-direction
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D




def create_board(num_pnts):
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

def trans_x_y(y_shift, x_shift, board):	
	x_shift_arr = np.full(num_pnts*num_pnts, x_shift, dtype = int) #array of 'shift's 
	y_shift_arr = np.full(num_pnts*num_pnts, y_shift, dtype = int) 


	boardtrans_x_y = np.array(board) #create new board so we don't modify original

	boardtrans_x_y[0,:]+=x_shift #add to x row
	boardtrans_x_y[1,:]+=y_shift #add to y row
	return boardtrans_x_y


def rotate(X, theta, axis='x'):
  '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
  c, s = np.cos(theta), np.sin(theta)
  if axis == 'x': return np.dot(np.array([
    [1.,  0,  0],
    [0 ,  c, -s],
    [0 ,  s,  c]
  ]), X)
  elif axis == 'y': return np.dot(np.array([
    [c,  0,  -s],
    [0,  1,   0],
    [s,  0,   c]
  ]), X)
  elif axis == 'z': return np.dot(X, np.array([
    [c, -s,  0 ],
    [s,  c,  0 ],
    [0,  0,  1.],
  ]), X)

def rot(board):
	#r1 = R.from_euler('x', 180, degrees=True)
	#print(r1.as_quat())
	
	board_rot = np.array(board)
	rot = rotate(board_rot[:3,:], 1.2, 'x')
	return rot

def euler_rot(angle, board):
	r1 = R.from_euler('x', angle, degrees=True)
	board_rot = np.array(board)
	return r1.apply( board_rot[:3,:])


def plot_board_3d(board,  ax, c, m):

	ax.scatter(board[0,:], board[1,:], board[2,:], 'z', c=c, marker=m)
	

def plot_board_2d(board, marker):
	plt.plot(board[0,:], board[1,:], marker)
	plt.grid(True)
		

if __name__ == '__main__':
	num_pnts = 5
	shift = 2
 	
	board = create_board(num_pnts)
	#print(board)
	
	boardtrans_x_y = trans_x_y(shift, shift, board)
	
	board_rot = rot(board)
	print(board_rot)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	plot_board_3d(board, ax, 'b', 'o')
	#plot_board_3d(boardtrans_x_y, ax, 'g', '>')
	plot_board_3d(board_rot,  ax, 'r', '*')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show()



