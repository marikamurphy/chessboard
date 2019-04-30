#simple plot of checkerboard points and entire board shifted +1 in y-direction
import numpy as np
import matplotlib.pyplot as plt



def create_board(num_pnts):
	board = np.full(num_pnts, 0, dtype = int) #this is x
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
	
	return board

def trans_x_y(y_shift, x_shift, board):	
	y_shift_arr = np.full(num_pnts*num_pnts, y_shift, dtype = int) #array of 'shift's
	x_shift_arr = np.full(num_pnts*num_pnts, x_shift, dtype = int) 

	boardtrans_x_y = np.array(board)#otherwise we modify board too 

	boardtrans_x_y[0,:]+=x_shift #add to x row
	boardtrans_x_y[1,:]+=y_shift #add to y row
	return boardtrans_x_y



if __name__ == '__main__':
	num_pnts = 5
	shift = 2

	board = create_board(num_pnts)
	
	boardtrans_x_y = trans_x_y(shift, shift, board)

	plt.plot(board[0,:], board[1,:], 'ro')
	plt.plot(boardtrans_x_y[0,:], boardtrans_x_y[1,:], 'b.')

	plt.axis([-1, num_pnts+shift+1, -1, num_pnts+shift+1]) # want axis to include whole board and shifted board

	plt.grid(True)
	plt.show()
