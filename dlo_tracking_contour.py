
# Import necessary libraries
import cv2
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt

def assignKeypoints(keypoints):
    keypoints_assigned = sorted([(x + w / 2, y + h / 2) for x, y, w, h in [cv2.boundingRect(cnt) for cnt in keypoints]], key=lambda x: x[0])
    return keypoints_assigned

def rescaleImg(frame, scale = 0.5):
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
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    # Create a mask
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate area for each contour and create a tuple (contour, area)
    contour_areas = [(cnt, cv2.contourArea(cnt)) for cnt in contours]

    # Sort contours based on area in descending order
    contour_areas.sort(key=lambda x: x[1], reverse=True)

    # Select top 11 contours
    top_area_contours = [cnt[0] for cnt in contour_areas[:11]]

    # Filter out smaller contours that are close to larger ones
    filtered_contours = []
    for cnt1 in top_area_contours:
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
    return copied_img, filtered_contours

#Read Video
video_path = 'videos\id17.mp4'
cap = cv2.VideoCapture(video_path)
if cap is not None:
    # Get the frames per second (fps) and frame size of the original video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(3)), int(cap.get(4)))

    # Create a cv2.VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # or use 'XVID'
    out = cv2.VideoWriter('output.avi', fourcc, fps , frame_size)
    keypoints_matrix = []
    while True:
        _ , img = cap.read()
        if img is not None:
            #Keypoint Alogorithms
            keypoints_vector = [0]*11
            img_with_color_contour, filtered_contours_by_area = colorContour(img)

            if keypoints_matrix:
                for cnt in filtered_contours_by_area:
                    short_dist, index = 100,0
                    x, y, w, h = cv2.boundingRect(cnt)
                    coordinates = (x + w / 2, y + h / 2)
                    for i,cnt_index in enumerate(keypoints_matrix[-1]):
                        if cnt_index != 0:
                            prev_coordinates = cnt_index
                        elif cnt_index == 0:
                            for j in range(1, 15):
                                cnt_index = keypoints_matrix[-j][i]
                                if cnt_index != 0:
                                    prev_coordinates = cnt_index
                                    break
                        ecludian_dist = ((coordinates[0] - prev_coordinates[0]) ** 2 + (coordinates[1] - prev_coordinates[1]) ** 2) ** 0.5
                        if ecludian_dist < short_dist:
                            index = i
                            short_dist = ecludian_dist
                    keypoints_vector[index] = coordinates

            else:
                keypoints_vector = assignKeypoints(filtered_contours_by_area)

            keypoints_matrix.append(keypoints_vector)
            
            """ keypoints_vector = np.array(keypoints_vector)
            plt.scatter(*zip(*keypoints_vector))
            plt.show() """

            #Rescale
            img_with_color_contour_resize = rescaleImg(img_with_color_contour)

            # Write the frame into the output video file
            out.write(img_with_color_contour_resize)

            # Display the image with keypoints
            cv2.imshow('Keypoints_colorContour', img_with_color_contour_resize)
            
            cv2.imshow('Video', rescaleImg(img))
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
        else:
            break
    # print (keypoints_matrix)

    with open('keypoints_matrix_id17.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(keypoints_matrix)

cap.release()
out.release()
cv2.destroyAllWindows()