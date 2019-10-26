import cv2
import os
from os import path

def video_to_frames(video_name):

    video_path = 'Input/' + video_name 
    cap = cv2.VideoCapture(video_path)
    output_path = 'Output/' + video_name;
    
    if not path.isdir(output_path):
        os.makedirs(output_path)

    count = 0

    while cap.isOpened():
        frameWasCaptured, frame = cap.read()
        if frameWasCaptured:
            cv2.imwrite(output_path + '/frame{:d}.jpg'.format(count), frame)
            count += 30 # i.e. at 30 fps, this advances one second
            cap.set(1, count)
        else:
            cap.release()
            break

input_path = 'Input'

for file in os.listdir(input_path):
    if file.endswith('.mp4'):
        video_to_frames(file)



