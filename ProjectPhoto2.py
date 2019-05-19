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

    #print(corners)
    print(chessImg.size)
    print(img3.size)
    x1 = int(round(corners[0,0]))
    y1 = int(round(corners[1,0]))
    x2 = int(round(corners[0,-7]))
    y2 = int(round(corners[1,-7]))
    print(x1)
    print(y1)
    print(x2)
    print(y2)


    
    # gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    # _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    # #cv2.imshow('thresh', thresh)
    # #cv2.waitKey(0)
    # contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[0]

    # x,y,w,h = cv2.boundingRect(cnt)
    # crop = img3[y:y+h,x:x+w]

    # thresh = np.array(thresh)
    # print("thresh")
    # print(thresh)
    # mask = thresh >0
    # crop = cv2.bitwise_and(img3, img3, mask=thresh)
    # not_mask = cv2.bitwise_not(thresh)
    # background = cv2.bitwise_and(chessImg,chessImg, mask = not_mask)
    # paste = cv2.bitwise_or(crop, background)


    gray = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    #cv2.imshow('thresh', thresh)
    #cv2.waitKey(0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    x,y,w,h = cv2.boundingRect(cnt)
    crop = img3[y:y+h,x:x+w]

    # tempimage = Image.fromarray(crop)
    # tempimage = tempimage.convert("RGBA")

    # datas = tempimage.getdata()

    # newData = []
    # for item in datas:
    #     if item[0] == 0 and item[1] == 0 and item[2] == 0:
    #         newData.append((255, 255, 255, 0))
    #     else:
    #         newData.append(item)

    # tempimage.putdata(newData)
    # tempimage.save("transparent_background.png", "PNG")

    img1 = np.array(chessImg)
    img2 = crop
    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = img2.shape
    roi = img1[y2:rows+y2, x2:cols+x2]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    print(roi)
    cv2.imshow('roi',roi)
    cv2.waitKey(0)
    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg,img2_fg)
    img1[y2:rows+y2, x2:cols+x2 ] = dst

    
    #transparent = mpimg.imread('transparent_background.jpg')
    # transparent = Image.open('transparent_background.png')
    #croppedImg3 = Image.fromarray(transparent)



    #   img3 = Image.fromarray(img3)
    # croppedImg3 = img3.crop((x1, y2, x2, y1))
    # w=np.array(croppedImg3)
    # print(w)

    # chessImg = chessImg.convert("RGBA")
    # print(chessImg.mode)
    # print(transparent.mode)
    # chessImg.paste(transparent,(x1, y1))
    # chessImg.save("chessimg.png", "PNG")
    # paste = np.array(chessImg)

    #print(paste)
    #chessImg = cv2.bitwise_or(chessImg, crop)
    return img1


# Make image and image matrices
img = mpimg.imread('logo.jpg')
img = img.copy()
img[np.where((img==[0,0,0]).all(axis=2))]=[2,2,2]
cam_img = mpimg.imread('board8.jpg')
img = makeSquare(img)

imgFLIP = img[::-1,:,:]

# Each is a matrix representing the x, y, or w value in the matrix
xMat, yMat, wMat, height, width = makeImgMat(img)

#Create original board
num_pnts = 7
#board3D = create_board3D(num_pnts, img)
#board2D = hom_3Dto2D(board3D)
board2D = create_board2D(num_pnts, img)
#Create rotated board
# rot_mat = random_trans_generator() #arbitrary rotation
# rigid_trans_mat = rigid_trans(rot_mat, [[0],[0], [0]])
# trans_mat = transform_matrix(board3D, rigid_trans_mat) #delete
# board2DTrans = hom_3Dto2D(trans_mat) #delete
chessImg, board2DTrans = getChessboardCorners(cam_img)
carB2DT = hom_cart_trans(board2DTrans) #delete
plt.subplot(2, 3, 1)
plt.plot(carB2DT[0,:], carB2DT[1,:], 'bo')
plt.grid(True)


#make a board2DTrans for every subsection
# put this into a loop

#find Homography and apply to original
hMat = getHomographyMat(board2D, board2DTrans)
x, y, w = makeTransformedImage(hMat, xMat, yMat, wMat, height, width)
print('X MATRIX')
print(x)
print('Y MATRIX')
print(y)
print('W MATRIX')
print(w)
#plot_board_2d(product, 'r*')
ax1 = plt.subplot(2, 3, 2)
ax1.imshow(imgFLIP, origin = 'lower')


ax2 = plt.subplot(2, 3, 3)
#img2 = cv2.remap(img, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_LINEAR) #how to fix lmao
ax2.imshow(chessImg, origin = 'lower')
ax3 = plt.subplot(2, 3, 4)
img_size = (width, height)
img_warped = cv2.warpPerspective(imgFLIP, hMat, img_size)
ax3.imshow(img_warped, origin = 'lower')

ax4 = plt.subplot(2, 3, 5)
pasted = copy_paste(chessImg, img_warped, board2DTrans)
ax4.imshow(pasted, origin = 'lower')
plt.show()




for r in range(0, num_pnts-1):
    for c in range(0, num_pnts-1):
        sub_board2D = board2D[r*num_pnts+c], board2D[r*num_pnts+(c+1)], board2D[(r+1)*num_pnts+c], board2D[(r+1)*num_pnts+(c+1)]
        sub_board2DTrans = 0 





# cv2.imshow('img3',img3)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()
#cv2.imshow('image',img2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


print('end')

