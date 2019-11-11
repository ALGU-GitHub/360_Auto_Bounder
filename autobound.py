import cv2
import os
import shutil
from os import path
import numpy as np
import imutils
import math 

def make_directory_if_missing(directory_name):
    if not path.isdir(directory_name):
        os.makedirs(directory_name)

def run_detection_on_image_at_path(local_path_to_image):
    image_name = os.path.splitext(local_path_to_image)[0]
    output_path = os.getcwd() + '/Output/' + image_name;
    make_directory_if_missing(output_path)
    
    angle_increment = 15
    image_from_path = cv2.imread(local_path_to_image)
    
     
    for angle in np.arange(0, 360, angle_increment):
        # Rotate current frame.
        rotated_frame = imutils.rotate(image_from_path, angle)
        
       
        
        new_image_path = output_path + '/' + image_name + '_{:d}.jpg'.format(angle)
        cv2.imwrite(new_image_path, rotated_frame)
        
        autobound_path = os.getcwd()
        darknet_path = 'darknet'
        os.chdir(darknet_path)
        os.system('./darknet detect cfg/yolov3.cfg cfg/yolov3.weights ' + new_image_path)
        prediction_path = os.getcwd() + '/predictions.jpg'
        new_prediction_path = output_path + '/' + image_name + '_{:d}_prediction.jpg'.format(angle)
        shutil.copy(prediction_path, new_image_path + '_prediction.jpg')
        os.remove(prediction_path)
        os.chdir(autobound_path)



def draw_boarders_around_frame_at(frame_path):
    frame = cv2.imread(frame_path)
    height, width, channels = frame.shape
    diagonal_length = int(math.hypot(height, width))
    horizontal_boarder = (diagonal_length - width)/2
    vertical_boarder = (diagonal_length - height)/2
    color = [110,100,100]
    top, bottom, left, right = [vertical_boarder, vertical_boarder, horizontal_boarder, horizontal_boarder]
    frame_with_border = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    cv2.imwrite(frame_path, frame_with_border)

def video_to_frames(video_name):
    count = 0
    
    video_path = 'Input/' + video_name 
    video_capture = cv2.VideoCapture(video_path)
    
    while video_capture.isOpened():
        frame_was_captured, frame = video_capture.read()
        if frame_was_captured:
            frame_path = video_name + '_frame{:d}.jpg'.format(count)
            cv2.imwrite(frame_path, frame)
            draw_boarders_around_frame_at(frame_path)
            run_detection_on_image_at_path(frame_path)
            count += 30 
            video_capture.set(1, count)
        else:
            video_capture.release()
            break



input_path = 'Input'
run_detection_on_image_at_path('Test.jpg')

#for file_in_input_path in os.listdir(input_path):
#    if file_in_input_path.endswith('.mp4'):
#        video_to_frames(file_in_input_path)

print('Done')
