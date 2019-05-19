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
    
    x1 = int(round(corners[0,0]))
    y1 = int(round(height-corners[1,0]))
    x2 = int(round(corners[0,-1]))
    y2 = int(round(height-corners[1,-1]))
    


    gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    x,y,w,h = cv2.boundingRect(cnt)
    crop = img3[y:y+h,x:x+w]

    tempimage = Image.fromarray(crop)
    tempimage = tempimage.convert("RGBA")

    datas = tempimage.getdata()

    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    tempimage.putdata(newData)
    


    chessImg.paste(tempimage,(x1, y1))
    paste = np.array(chessImg)

    return paste

def findIndHoms(board2D,board2DTrans, num_pnts):
	homs = np.array(getHomographyMat(board2D[:,0:2:num_pnts],board2DTrans[:,0:2:num_pnts]))
	numP = num_pnts**2
	i = 1

	while i < numP-num_pnts:
		board1 = np.concatenate((board2D[:,i:i+2], board2D[:,i+num_pnts:i+2+num_pnts]),axis = 1)
		board2 = np.concatenate((board2DTrans[:,i:i+2], board2DTrans[:,i+num_pnts:i+2+num_pnts]),axis = 1)
		print(board1)
		homs = np.concatenate((homs,getHomographyMat(board1,board2)),axis=0)
		i= i+1
		
		if i%(num_pnts-1) == 0:
			i = i+1
	homs = np.reshape(np.array(homs),((num_pnts-1)**2,3,3))
	return homs	

def indHomTrans(imgFLIP,board2D, homs, w, h,num_pnts):
	print(board2D)
	
	
	numP = 0
	x = 0
	while x < homs.shape[0]-num_pnts+2:
		for i in range(0,num_pnts-1):
			x1 = int(board2D[0,i])
			x2 = int(board2D[0,i+1])
			y1 = int(board2D[1,i+numP])
			y2 = int(board2D[1,i+numP+num_pnts])
			ax4 = plt.subplot(6, 6, x+1)
			print(y1)
			print(y2)
			im = imgFLIP[y2:y1,x1:x2]
			img4 = cv2.warpPerspective(im, homs[x], (w,h))
			ax4.imshow(img4, origin = 'lower')
			x= x+1
		numP= numP+num_pnts
			
	
	cv2.imshow('img3',img4)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# Make image and image matrices
img = mpimg.imread('logo.jpg')
img = img.copy()
img[np.where((img==[0,0,0]).all(axis=2))]=[2,2,2]
cam_img = mpimg.imread('board8.jpg')
img = makeSquare(img)

img2 = Image.fromarray(img)
img2 = img2.convert("RGBA")

datas = img2.getdata()

newData = []
for item in datas:
    if item[0] == 0 and item[1] == 0 and item[2] == 0:
        newData.append((255, 255, 255, 0))
    else:
        newData.append(item)

img2.putdata(newData)
img2.save("img2.png", "PNG")

imgFLIP = img[::-1,:,:]

# Each is a matrix representing the x, y, or w value in the matrix
xMat, yMat, wMat, height, width = makeImgMat(img)

#Create original board
num_pnts = 7

board2D = create_board2D(num_pnts, img)

chessImg, board2DTrans = getChessboardCorners(cam_img)
carB2DT = hom_cart_trans(board2DTrans) #delete
plt.subplot(6, 6, 1)
plt.plot(carB2DT[0,:], carB2DT[1,:], 'bo')
plt.plot(carB2DT[0,0], 'ro')
plt.grid(True)


#make a board2DTrans for every subsection
# put this into a loop
homs = findIndHoms(board2D, board2DTrans, num_pnts)
#find Homography and apply to original
hMat1 = getHomographyMat(board2D[:,:16], board2DTrans[:,:16])
hMat2 = getHomographyMat(board2D[:,16:], board2DTrans[:,16:])

ax1 = plt.subplot(6, 6, 2)
ax1.imshow(imgFLIP, origin = 'lower')


ax2 = plt.subplot(6, 6, 3)
#img2 = cv2.remap(img, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_LINEAR) #how to fix lmao
ax2.imshow(chessImg, origin = 'lower')
ax3 = plt.subplot(6, 6, 4)
img_size = (width, height)
img3 = cv2.warpPerspective(imgFLIP, hMat1, img_size)
ax3.imshow(img3, origin = 'lower')

indHomTrans(imgFLIP, board2D, homs, width,height,num_pnts)
'''img4 = cv2.warpPerspective(imgFLIP[100:100+int(width/(num_pnts-2)),100:100+int(height/(num_pnts-2))], homs[2], (width,height))
ax4 = plt.subplot(2, 3, 5)
ax4.imshow(img4, origin = 'lower')'''
plt.show()
# cv2.imshow('img3',img3)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()
#cv2.imshow('image',img2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


print('end')

