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
    detections_path =  os.getcwd() + '/Detections'
    make_directory_if_missing(detections_path)
    
    output_path = os.getcwd() + '/Output/' + local_path_to_image;
    make_directory_if_missing(output_path)
    
    angle_increment = 15
    image_from_path = cv2.imread(local_path_to_image)
    
    for angle in np.arange(0, 180, angle_increment):
        rotated_frame = imutils.rotate_bound(image_from_path, angle)
        new_image_path = local_path_to_image + '_{:d}.jpg'.format(angle)
        cv2.imwrite(new_image_path, rotated_frame)
        autobound_path = os.getcwd()
        true_path_to_image = os.getcwd() + '/' + new_image_path
        darknet_path = 'darknet'
        os.chdir(darknet_path)
        os.system('./darknet detect cfg/yolov3.cfg cfg/yolov3.weights ' + true_path_to_image)
        prediction_path = os.getcwd() + '/predictions.jpg'
        shutil.copy(prediction_path, detections_path + '/' + new_image_path + '_prediction.jpg')
        os.remove(prediction_path)
        os.chdir(autobound_path)
        shutil.copy(new_image_path, output_path)
        os.remove(new_image_path)



def video_to_frames(video_name):
    count = 0
    
    video_path = 'Input/' + video_name 
    video_capture = cv2.VideoCapture(video_path)
    
    while video_capture.isOpened():
        frame_was_captured, frame = video_capture.read()
        if frame_was_captured:
            frame_path = video_name + '_frame{:d}.jpg'.format(count)
            cv2.imwrite(frame_path, frame)
            run_detection_on_image_at_path(frame_path)
            count += 30 
            video_capture.set(1, count)
        else:
            video_capture.release()
            break



input_path = 'Input'
#run_detection_on_image_at_path('frame270_0.jpg')

for file_in_input_path in os.listdir(input_path):
    if file_in_input_path.endswith('.mp4'):
        video_to_frames(file_in_input_path)

print('Done')
