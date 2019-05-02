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
def trans_point(X, x_shift=0, y_shift=0, z_shift=0):
	X[0]+=x_shift
	X[1]+=y_shift
	X[2]+=z_shift
	return X

def trans_x_y_z(board, x_shift=0, y_shift=0, z_shift=0):	
	
	boardtrans_x_y = np.array(board) #create new board so we don't modify original

	boardtrans_x_y[0,:]+=x_shift #add to x row
	boardtrans_x_y[1,:]+=y_shift #add to y row
	boardtrans_x_y[2,:]+=z_shift #add to y row
	return boardtrans_x_y

def rotate_point(X, theta, axis='x'):
	'''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
	c, s = np.cos(theta), np.sin(theta)
	if axis == 'x': X = np.dot(np.array([
    		[1.,  0,  0],
    		[0 ,  c, -s],
    		[0 ,  s,  c]
  		]), X)
	elif axis == 'y': X = np.dot(np.array([
    		[c,  0,  -s],
    		[0,  1,   0],
    		[s,  0,   c]
  		]), X)
	elif axis == 'z': X = np.dot(X, np.array([
    		[c, -s,  0 ],
    		[s,  c,  0 ],
    		[0,  0,  1.],
  		]), X)
	return X

def rotate_board(board, theta, axis='x'):
	'''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
	board_rot = np.array(board)
	c, s = np.cos(theta), np.sin(theta)
	if axis == 'x': return np.dot(np.array([
    		[1.,  0,  0],
    		[0 ,  c, -s],
    		[0 ,  s,  c]
  		]), board_rot[:3,:])
	elif axis == 'y': return np.dot(np.array([
    		[c,  0,  -s],
    		[0,  1,   0],
    		[s,  0,   c]
  		]), board_rot[:3,:])
	elif axis == 'z': return np.dot(np.array([
    		[c, -s,  0 ],
    		[s,  c,  0 ],
    		[0,  0,  1.],
  		]), board_rot[:3,:])


def plot_board_3d(board,  ax, c, m):

	ax.scatter(board[0,:], board[1,:], board[2,:], 'z', c=c, marker=m)
	

def plot_board_2d(board, marker):
	plt.plot(board[0,:], board[1,:], marker)
	plt.grid(True)

		

if __name__ == '__main__':

	#this is all just me testing stuff
	num_pnts = 5
	shift = 2
 	
	board = create_board(num_pnts)
	#print(board)
	
	boardtrans_x_y = trans_x_y_z(board,shift, shift)
	boardtrans_x_y[:3,1] = trans_point(boardtrans_x_y[:3,1],shift, shift,shift)

	print(boardtrans_x_y)
	board_rot = rotate_board(board, 1.2)
	#print(board_rot[:3,1])
	board_rot[:3,1] = rotate_point(board_rot[:3,1], 1.2, 'x')
	board_rot[:3,2] = rotate_point(board_rot[:3,2], 1.2, 'x')
	board_rot[:3,5] = rotate_point(board_rot[:3,1], 1.2, 'y')
	board_rot[:3,6] = rotate_point(board_rot[:3,2], 1.2, 'y')
	#print(board_rot)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	plot_board_3d(board, ax, 'b', 'o')
	plot_board_3d(boardtrans_x_y, ax, 'g', '>')
	#plot_board_3d(board_rot,  ax, 'r', '*')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	plt.show()



