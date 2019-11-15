import cv2
import os
import shutil
from os import path
import numpy as np
import imutils
import math 

class BoundData:
    def __init__(self):
        self.object_class = 0
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.box_width = 0.0
        self.box_height = 0.0
    
    def bound_info_string_to_variables(self, bound_info_string):
        # <object-class> <x> <y> <width> <height>
        values = map(float, bound_info_string.split(" "))
        self.object_class = values[0]
        self.x_pos = values[1]
        self.y_pos = values[2]
        self.box_width = values[3]
        self.box_height = values[4]
        
def calculate_angle_difference_in_radians(angle_to, angle_from):
    angle_difference_in_radians = 0
    if angle_from < angle_to:
        angle_difference_in_radians = math.radians((angle_from + 360) - angle_to)
    else:
         angle_difference_in_radians = math.radians(angle_from - angle_to)
    return angle_difference_in_radians
 
def calculate_rotated_position(x_pos, y_pos, x_origin, y_origin, angle_difference_in_radians):
        calibrated_x = x_pos - x_origin
        calibrated_y = y_pos - y_origin
        rotated_x_pos = (calibrated_x * math.cos(angle_difference_in_radians)) - (calibrated_y * math.sin(angle_difference_in_radians)) + x_origin
        rotated_y_pos = (calibrated_x * math.sin(angle_difference_in_radians)) + (calibrated_y * math.cos(angle_difference_in_radians)) + y_origin
        return rotated_x_pos, rotated_y_pos

def calculate_rotated_dimensions(box_width, box_height, angle_difference_in_radians):
    rotated_box_width = 0
    rotated_box_height = 0
    congu = angle_difference_in_radians % (math.pi)
    if  congu > math.pi:
        angle_difference_in_radians = angle_difference_in_radians - math.pi
        rotated_box_width = (box_height * math.cos(angle_difference_in_radians)) + (box_width * math.sin(angle_difference_in_radians)) 
        rotated_box_height = (box_height * math.sin(angle_difference_in_radians)) + (box_width * math.cos(angle_difference_in_radians)) 
    else:
        rotated_box_width = (box_width * math.cos(angle_difference_in_radians)) + (box_height * math.sin(angle_difference_in_radians)) 
        rotated_box_height = (box_width * math.sin(angle_difference_in_radians)) + (box_height * math.cos(angle_difference_in_radians)) 
    return abs(rotated_box_width), abs(rotated_box_height)

def debug_bound_info(image_directory, angle_increment):
   
    for current_angle in np.arange(0, 360, angle_increment):
        new_image_path = image_directory + '_{:d}.jpg'.format(current_angle)
        new_bound_info_path = image_directory + '_{:d}.txt'.format(current_angle)
        print new_image_path
        with open(new_bound_info_path) as bound_info_file:
            bound_info_string = bound_info_file.readline()
            while bound_info_string:
                current_bound_data = BoundData()
                current_bound_data.bound_info_string_to_variables(bound_info_string)
                image = cv2.imread(new_image_path, cv2.IMREAD_COLOR)
                cv2.rectangle(image,(366,345),(40,522),(0,255,0),3)
                cv2.imshow('debug_bounds_' + str(current_angle), image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                bound_info_string = bound_info_file.readline()

def make_directory_if_missing(directory_name):
    if not path.isdir(directory_name):
        os.makedirs(directory_name)

def run_detection_on_image_at_path(local_path_to_image):

    image_name = os.path.splitext(local_path_to_image)[0]
    output_path = os.getcwd() + '/Output/' + image_name;
    make_directory_if_missing(output_path)
    
    angle_increment = 45
    image_from_path = cv2.imread(local_path_to_image)
    
    list_of_files = []
    
    # Make text files that will hold the bounding info for each rotation.
    for angle in np.arange(0, 360, angle_increment):
        new_bound_info_path = output_path + '/' + image_name + '_{:d}.txt'.format(angle)
        new_bound_info_file = open(new_bound_info_path, "a+")
        new_bound_info_file.truncate(0)
        list_of_files.append(new_bound_info_file)
    
    for current_angle in np.arange(0, 360, angle_increment):
        # Rotate current frame and save said rotated frame.
        rotated_frame = imutils.rotate(image_from_path, current_angle)
        new_image_path = output_path + '/' + image_name + '_{:d}.jpg'.format(current_angle)
        cv2.imwrite(new_image_path, rotated_frame)
        
        rotated_frame_image = cv2.imread(new_image_path)
        rotated_frame_height, rotated_frame_width, rotated_frame_channels = rotated_frame_image.shape
        rotated_frame_x_origin = rotated_frame_width / 2
        rotated_frame_y_origin = rotated_frame_height / 2
        
        # Run Detection on said rotated frame.
        autobound_path = os.getcwd()
        darknet_path = 'darknet'
        os.chdir(darknet_path)
        #print
        os.system('./darknet detect cfg/yolov3.cfg cfg/yolov3.weights ' + new_image_path)
        
        
        bound_info_path = autobound_path + '/Output.txt'
        with open(bound_info_path) as bound_info_file:
            bound_info_string = bound_info_file.readline()
            while bound_info_string:
            
                current_bound_data = BoundData()
                current_bound_data.bound_info_string_to_variables(bound_info_string)
                file_counter = 0
                for angle in np.arange(0, 360, angle_increment):
                    angle_difference_in_radians = calculate_angle_difference_in_radians(angle, current_angle)
                    rotated_x_pos, rotated_y_pos = calculate_rotated_position(current_bound_data.x_pos, current_bound_data.y_pos, rotated_frame_x_origin, rotated_frame_y_origin, angle_difference_in_radians)
                    rotated_box_width, rotated_box_height = calculate_rotated_dimensions(current_bound_data.box_width, current_bound_data.box_height, angle_difference_in_radians)
                    #print('angle : ' + str(current_angle) + ' --> ' +str(angle))
                    bounding_box_info_to_be_written = '0 ' + str(rotated_x_pos/rotated_frame_width) + ' ' + str(rotated_y_pos/rotated_frame_height) + ' ' + str(rotated_box_width/rotated_frame_width)  + ' ' + str(rotated_box_height/rotated_frame_height)
                    #print(bounding_box_info_to_be_written)
                    list_of_files[file_counter].write(bounding_box_info_to_be_written + '\n')
                    file_counter += 1
                bound_info_string = bound_info_file.readline()
                
        new_prediction_path = output_path + '/' + image_name + '_{:d}_prediction.jpg'.format(current_angle)
        prediction_path = os.getcwd() + '/predictions.jpg'
        shutil.copy(prediction_path, new_image_path + '_prediction.jpg')
        os.remove(prediction_path)
        os.chdir(autobound_path)
        
    for file in list_of_files:
        file.close()
    del list_of_files[:]
    
    image_directory = output_path + '/' + image_name
    debug_bound_info(image_directory, angle_increment)

    
     
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
