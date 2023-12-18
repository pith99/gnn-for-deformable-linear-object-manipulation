import numpy as np
from scipy import signal
import cv2

def rescaleImg(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


img = cv2.imread('images\S1\JPEGImages\img6.jpeg')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow('img',img)

bg = signal.medfilt2d(img, 11)
mask1 = img < bg - 70
mask1 = np.where(mask1, img, 0.0)



mask_display1 = (mask1.astype(np.uint8) * 255).reshape(mask1.shape + (1,))

mask_display1_resize = rescaleImg(mask_display1, scale = 0.5)



cv2.imshow('mask', mask_display1_resize)


# # Initialize the ORB detector
# orb = cv2.ORB_create()

# # Detect keypoints
# keypoints = orb.detect(img, None)

# # Draw keypoints on the image
# img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)

# # Display the image with keypoints
# cv2.imshow('Keypoints', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
