
# Import necessary libraries
import cv2
import numpy as np


def orb_detector(img):
    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints
    keypoints = orb.detect(img, None)

    # Draw keypoints on the image
    return cv2.drawKeypoints(img, keypoints, None, color=(0,255,0), flags=0)

def rescaleImg(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

def goodFeaturesToTrack(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),5,255,-1)
    return img      

#Read Image
img = cv2.imread('images\S1\JPEGImages\img.jpeg')

#Keypoint Alogorithms
img_with_keypoints_orb = orb_detector(img)
img_with_keypoints_goodFeaturesToTrack = goodFeaturesToTrack(img)

#Rescale
img_with_keypoints_orb_resize = rescaleImg(img_with_keypoints_orb, scale = 0.5)
img_with_keypoints_goodFeaturesToTrack_resize = rescaleImg(img_with_keypoints_goodFeaturesToTrack, scale = 0.5)

# Display the image with keypoints
cv2.imshow('Keypoints', img_with_keypoints_orb_resize)
cv2.imshow('Keypoints_goodFeaturesToTrack', img_with_keypoints_goodFeaturesToTrack_resize)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()