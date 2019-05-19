import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


def padImg(img, border_size):
	color = [0, 0, 0]
	img_with_border = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=color)
	return img_with_border

def makeSquare(img):
	color = [0, 0, 0]
	dimensions = img.shape
    # height, width, number of channels in image
	height = dimensions[0]
	width = dimensions[1]
	padding = 0
	if height>width:
		temp = (height - width)/2
		padding = round(temp)
		img_with_border = cv2.copyMakeBorder(img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=color)
	else:
		temp = (width - height)/2
		padding = round(temp)
		img_with_border = cv2.copyMakeBorder(img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=color)

	return img_with_border

