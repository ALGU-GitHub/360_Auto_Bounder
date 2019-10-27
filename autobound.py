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

    count = 0

    while video_capture.isOpened():
        frameWasCaptured, frame = video_capture.read()
        if frameWasCaptured:
            for angle in np.arange(0, 30, 15):
                rotated_frame = imutils.rotate_bound(frame, angle)     
                cv2.imwrite(output_path + '/frame{:d}_{:d}.jpg'.format(count, angle), rotated_frame)
            count += 200 # i.e. at 30 fps, this advances one second
            video_capture.set(1, count)
        else:
            video_capture.release()
            break

input_path = 'Input'

for file in os.listdir(input_path):
    if file.endswith('.mp4'):
        video_to_frames(file)



