
# Import necessary libraries
import cv2
import numpy as np

def rescaleImg(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Function to check if two contours are close
def are_contours_close(cnt1, cnt2, threshold=10):
    x1, y1, w1, h1 = cv2.boundingRect(cnt1)
    x2, y2, w2, h2 = cv2.boundingRect(cnt2)

    # Calculate the distance between the centers of the bounding rectangles
    distance = ((x1 + w1 / 2 - x2 - w2 / 2) ** 2 + (y1 + h1 / 2 - y2 - h2 / 2) ** 2) ** 0.5
    return distance < threshold

def colorContour(img):
    # Convert BGR image to HSV
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define the range for white color
    lower_red = np.array([160, 50, 50])
    upper_red = np.array([180, 255, 255])
    # Create a mask
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Calculate area for each contour and create a tuple (contour, area)
    contour_areas = [(cnt, cv2.contourArea(cnt)) for cnt in contours]

    # Sort contours based on area in descending order
    contour_areas.sort(key=lambda x: x[1], reverse=True)

    # Select top 20 contours
    top_area_contours = [cnt[0] for cnt in contour_areas[:11]]

    # Filter out smaller contours that are close to larger ones
    filtered_contours = []
    for i, cnt1 in enumerate(top_area_contours):
        keep = True
        for cnt2 in top_area_contours:
            if cnt2 is not cnt1:
                if are_contours_close(cnt1, cnt2) and cv2.contourArea(cnt1) < cv2.contourArea(cnt2):
                    keep = False
                    break
        if keep:
            filtered_contours.append(cnt1)

    # Draw bounding box around color block
    copied_img = img.copy()
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(copied_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Draw a cross marker
        point = (int(x+w/2), int(y+h/2))
        cv2.drawMarker(copied_img, point, color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

    return copied_img

#Read Video
cap = cv2.VideoCapture('videos\id3.mp4')

# Get the frames per second (fps) and frame size of the original video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(3)), int(cap.get(4)))

# Create a cv2.VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # or use 'XVID'
out = cv2.VideoWriter('output.avi', fourcc, fps , frame_size)

while True:
    _ , img = cap.read()

    #Keypoint Alogorithms
    img_with_color_contour = colorContour(img)

    #Rescale
    img_with_color_contour_resize = rescaleImg(img_with_color_contour)

    # Write the frame into the output video file
    out.write(img_with_color_contour)

    # Display the image with keypoints
    cv2.imshow('Keypoints_colorContour', img_with_color_contour)
    
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


# #Read Image
# img = cv2.imread('images\S1\JPEGImages\img10.jpeg')

# #Keypoint Alogorithms
# img_with_color_contour = colorContour(img)

# #Rescale
# img_with_color_contour_resize = rescaleImg(img_with_color_contour)
# # Display the image with keypoints
# cv2.imshow('Keypoints_colorContour', img_with_color_contour_resize)

cap.release()
out.release()
cv2.destroyAllWindows()