#simple plot of checkerboard points and entire board shifted +1 in y-direction
import numpy as np
import matplotlib.pyplot as plt

num_pnts = 5


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
	
y_shift = np.full(num_pnts*num_pnts, 1, dtype = int) #array of 1s


boardtransy = np.array(board)#otherwise we modify board too 

boardtransy[1,:]+=y_shift #add to y row
print(board)
print(boardtransy)

plt.plot(board[0,:], board[1,:], 'ro')
plt.plot(boardtransy[0,:], boardtransy[1,:], 'b.')
plt.axis([-1, num_pnts+1, -1, num_pnts+1])
plt.grid(True)

plt.show()
