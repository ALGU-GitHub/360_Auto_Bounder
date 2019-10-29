import cv2
import os
import shutil
from os import path
import numpy as np
import imutils

def make_directory_if_missing(directory_name):
    if not path.isdir(directory_name):
        os.makedirs(directory_name)

def run_detection_on_image_at_path(local_path_to_image):
    autobound_path = os.getcwd()
    true_path_to_image = os.getcwd() + '/' + local_path_to_image
    darknet_path = 'darknet'
    os.chdir(darknet_path)
    os.system('./darknet detect cfg/yolov3.cfg cfg/yolov3.weights ' + true_path_to_image)
    prediction_path = os.getcwd() + '/predictions.jpg'
    shutil.copy(prediction_path, true_path_to_image + '_prediction')
    os.chdir(autobound_path)


def video_to_frames(video_name):

    output_path = 'Output/' + video_name;
    make_directory_if_missing(output_path)

    detections_path = output_path + '/Detections'
    make_directory_if_missing(detections_path)

    count = 0
    angle_increment = 15
    
    video_path = 'Input/' + video_name 
    video_capture = cv2.VideoCapture(video_path)
    
    while video_capture.isOpened():
        frame_was_captured, frame = video_capture.read()
        if frame_was_captured:
            #cv2.imwrite(output_path + '/*frame{:d}.jpg'.format(count), frame)
            for angle in np.arange(0, 15, angle_increment):
                rotated_frame = imutils.rotate_bound(frame, angle)
                new_image_path = output_path + '/frame{:d}_{:d}.jpg'.format(count, angle)
                cv2.imwrite(new_image_path, rotated_frame)
                run_detection_on_image_at_path(new_image_path)
            count += 30 # i.e. at 30 fps, this advances one second
            video_capture.set(1, count)
        else:
            video_capture.release()
            break

input_path = 'Input'

for file_in_input_path in os.listdir(input_path):
    if file_in_input_path.endswith('.mp4'):
        video_to_frames(file_in_input_path)
print('Done')
