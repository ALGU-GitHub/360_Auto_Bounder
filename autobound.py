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
    
    list_of_files = []
    
    # Make text files that will hold the bounding info for each rotation.
    for angle in np.arange(0, 360, angle_increment):
        new_bound_info_path = output_path + '/' + image_name + '_{:d}.txt'.format(angle)
        new_bound_info_file = open(new_bound_info_path, "w")
        list_of_files.append(new_bound_info_file)
        
    for current_angle in np.arange(0, 360, angle_increment):
        # Rotate current frame and save said rotated frame.
        rotated_frame = imutils.rotate(image_from_path, current_angle)
        new_image_path = output_path + '/' + image_name + '_{:d}.jpg'.format(current_angle)
        cv2.imwrite(new_image_path, rotated_frame)
        
        # Run Detection on said rotated frame.
        autobound_path = os.getcwd()
        darknet_path = 'darknet'
        os.chdir(darknet_path)
        os.system('./darknet detect cfg/yolov3.cfg cfg/yolov3.weights ' + new_image_path)
        
        
        bound_info_path = autobound_path + '/Output.txt'
        with open(bound_info_path) as bound_info_file:
            line = bound_info_file.readline()
            while line:
                # <object-class> <x> <y> <width> <height>
                values = map(int, line.split(" "))
                x_pos = values[1]
                y_pos = values[2]
                box_width = values[3]
                box_height = values[4]
               
                
                for angle in np.arange(0, 360, angle_increment):
                    angle_difference_in_radians = 0
                    if angle < current_angle:
                        angle_difference_in_radians = math.radians((angle + 360) - current_angle)
                    else:
                         angle_difference_in_radians = math.radians(angle - current_angle)
                    rotated_x_pos = (x_pos * math.cos(angle_difference_in_radians)) - (y_pos * math.sin(angle_difference_in_radians))
                    rotated_y_pos = (x_pos * math.sin(angle_difference_in_radians)) + (y_pos * math.cos(angle_difference_in_radians))
                       
                line = bound_info_file.readline()
                
        new_prediction_path = output_path + '/' + image_name + '_{:d}_prediction.jpg'.format(current_angle)
        prediction_path = os.getcwd() + '/predictions.jpg'
        shutil.copy(prediction_path, new_image_path + '_prediction.jpg')
        os.remove(prediction_path)
        os.chdir(autobound_path)

    for file in list_of_files:
        file.close()
        
    list_of_files.clear()
    
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
