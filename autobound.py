import cv2
import os
from os import path
import numpy as np
import imutils

def video_to_frames(video_name):

    video_path = 'Input/' + video_name 
    video_capture = cv2.VideoCapture(video_path)
    
    output_path = 'Output/' + video_name;
    
    if not path.isdir(output_path):
        os.makedirs(output_path)
    
    unaltered_output_path = output_path + '/Unaltered Frames'
    
    if not path.isdir(unaltered_output_path):
        os.makedirs(unaltered_output_path)
        
    altered_output_path = output_path + '/Altered Frames'
    
    if not path.isdir(altered_output_path):
        os.makedirs(altered_output_path)


    count = 0
    angle_increment = 15
    
    while video_capture.isOpened():
        frameWasCaptured, frame = video_capture.read()
        if frameWasCaptured:
            cv2.imwrite(unaltered_output_path + '/*frame{:d}.jpg'.format(count), frame)
            for angle in np.arange(angle_increment, 30, angle_increment):
                rotated_frame = imutils.rotate_bound(frame, angle)     
                cv2.imwrite(altered_output_path + '/frame{:d}_{:d}.jpg'.format(count, angle), rotated_frame)
            count += 30 # i.e. at 30 fps, this advances one second
            video_capture.set(1, count)
        else:
            video_capture.release()
            break

input_path = 'Input'

for file in os.listdir(input_path):
    if file.endswith('.mp4'):
        video_to_frames(file)



