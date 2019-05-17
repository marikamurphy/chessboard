import cv2
def padImg(img, border_size):
	color = [0, 0, 0]
	img_with_border = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=color)
	return img_with_border


