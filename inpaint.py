
import numpy as np
import cv2

img = cv2.imread('data\\20.jpg')
mask = cv2.imread('data\\mask.jpg',0)

dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
cv2.namedWindow('dst',0)
cv2.imwrite('data\\masked_20.jpg',dst)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()