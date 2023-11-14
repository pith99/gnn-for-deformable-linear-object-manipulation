import cv2
import numpy as np

img = cv2.imread('\images\S1\JPEGImages\image1.jpeg')
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
