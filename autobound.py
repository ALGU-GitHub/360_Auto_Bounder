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
        self.image_width = 0
        self.image_height = 0
        
    def bound_info_string_to_variables(self, bound_info_string):
        # <object-class> <x> <y> <width> <height>
        values = map(float, bound_info_string.split(" "))
        self.object_class = values[0]
        self.x_pos = values[1]
        self.y_pos = values[2]
        self.box_width = values[3]
        self.box_height = values[4]
        
        
    def set_height(self, new_height):
        self.image_height = new_height
        
    def set_width(self, new_width):
        self.image_width = new_width
    
    def get_local_x_pos(self):
        return self.x_pos * self.image_width
        
    def get_local_y_pos(self):
        return self.y_pos * self.image_height
        
    def distance_to_other_bound_data(self, other_bound_data):
        x_diff_squared = (self.x_pos - (self.image_width * other_bound_data.x_pos)) ** 2
        y_diff_squared =(self.y_pos - (self.image_height * other_bound_data.y_pos)) ** 2
        distance = math.sqrt(x_diff_squared + y_diff_squared)
        return distance
        
    def would_be_redundant_in(self, bound_info_path):
        if os.stat(bound_info_path).st_size == 0: 
            return False
        else:
            with open(bound_info_path) as bound_info_file:
                bound_info_string = bound_info_file.readline()
                while bound_info_string:         
                    bound_data_to_compare = BoundData()
                    bound_data_to_compare.bound_info_string_to_variables(bound_info_string)

                    if self.distance_to_other_bound_data(bound_data_to_compare) <= 25 :
                        bound_info_file.close()
                        return True
                    bound_info_string = bound_info_file.readline()
            bound_info_file.close()
            return False  
        
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

    congu = angle_difference_in_radians % (math.pi)

    if  congu > (math.pi/2):
        angle_difference_in_radians = math.pi - angle_difference_in_radians 
        rotated_box_width = (box_height * math.cos(angle_difference_in_radians)) + (box_width * math.sin(angle_difference_in_radians)) 
        rotated_box_height = (box_height * math.sin(angle_difference_in_radians)) + (box_width * math.cos(angle_difference_in_radians)) 
        return abs(rotated_box_width), abs(rotated_box_height)
    else:
        rotated_box_width = (box_width * math.cos(angle_difference_in_radians)) + (box_height * math.sin(angle_difference_in_radians)) 
        rotated_box_height = (box_width * math.sin(angle_difference_in_radians)) + (box_height * math.cos(angle_difference_in_radians)) 
        return abs(rotated_box_width), abs(rotated_box_height)

def make_directory_if_missing(directory_name):
    if not path.isdir(directory_name):
        os.makedirs(directory_name)


def debug_bound_info(output_path, image_name, angle_increment, frame_number):
    image_directory = output_path + '/' + image_name
    boxes_directory = 'Output/' + image_name + '/Boxes'
    make_directory_if_missing(boxes_directory)
    
    for current_angle in np.arange(0, 360, angle_increment):
        new_image_path = image_directory + '_f{:d}_a{:d}.jpg'.format(frame_number, current_angle)
        new_bound_info_path = image_directory + '_f{:d}_a{:d}.txt'.format(frame_number, current_angle)
        if os.stat(new_bound_info_path).st_size != 0:
            with open(new_bound_info_path) as bound_info_file:
                bound_info_string = bound_info_file.readline()
                image = cv2.imread(new_image_path, cv2.IMREAD_COLOR)
                while bound_info_string:
                    current_bound_data = BoundData()
                    current_bound_data.bound_info_string_to_variables(bound_info_string)
                    height, width, channels = image.shape
                    p1_x  = int(width * (current_bound_data.x_pos - (current_bound_data.box_width/2)))
                    p1_y = int(height * (current_bound_data.y_pos - (current_bound_data.box_height/2)))
                    p2_x  = int(width * (current_bound_data.x_pos + (current_bound_data.box_width/2)))
                    p2_y = int(height * (current_bound_data.y_pos + (current_bound_data.box_height/2)))
                    cv2.rectangle(image,(p1_x,p1_y),(p2_x,p2_y),(0,255,0),3)
                    bound_info_string = bound_info_file.readline()
            new_boxes_path = boxes_directory + '/' + image_name + '_f{:d}_a{:d}_boxes.jpg'.format(frame_number, current_angle)
            cv2.imwrite(new_boxes_path, image)
        else:
            os.remove(new_image_path)
            os.remove(new_bound_info_path)



def run_detection_on_frame(current_frame, image_name, frame_number):

    output_path = os.getcwd() + '/Output/' + image_name;
    make_directory_if_missing(output_path)
    
    angle_increment = 15

    # Make text files that will hold the bounding info for each rotation.
    for angle in np.arange(0, 360, angle_increment):
        new_bound_info_path = output_path + '/' + image_name + '_f{:d}_a{:d}.txt'.format(frame_number, angle)
        new_bound_info_file = open(new_bound_info_path, "a+")
        new_bound_info_file.truncate(0)
        new_bound_info_file.close()
    
    for current_angle in np.arange(0, 360, angle_increment):
        # Rotate current frame and save said rotated frame.
        rotated_frame = imutils.rotate(current_frame, current_angle)
        new_image_path = output_path + '/' + image_name + '_f{:d}_a{:d}.jpg'.format(frame_number, current_angle)
        
        cv2.imwrite(new_image_path, rotated_frame)
        
        rotated_frame_image = cv2.imread(new_image_path)
        rotated_frame_height, rotated_frame_width, rotated_frame_channels = rotated_frame.shape
        rotated_frame_x_origin = rotated_frame_width / 2
        rotated_frame_y_origin = rotated_frame_height / 2
        
        # Run Detection on said rotated frame.
        autobound_path = os.getcwd()
        darknet_path = autobound_path + '/darknet'
        os.chdir(darknet_path)

        os.system('./darknet detect cfg/yolov3.cfg cfg/yolov3.weights ' + new_image_path + '  -thresh 0.6')
        
        
        bound_info_path = autobound_path + '/Output.txt'
        with open(bound_info_path) as bound_info_file:
            bound_info_string = bound_info_file.readline()
            while bound_info_string:         
                current_bound_data = BoundData()
                current_bound_data.bound_info_string_to_variables(bound_info_string)
                current_bound_data.set_height(rotated_frame_height)
                current_bound_data.set_width(rotated_frame_width)
                current_bound_info_path = autobound_path + '/Output/' + image_name + '/' + image_name + '_f{:d}_a{:d}.txt'.format(frame_number, current_angle)
                

                
                if not current_bound_data.would_be_redundant_in(current_bound_info_path):
                    for angle in np.arange(0, 360, angle_increment):
                        angle_difference_in_radians = calculate_angle_difference_in_radians(angle, current_angle)
                        rotated_x_pos, rotated_y_pos = calculate_rotated_position(current_bound_data.x_pos, current_bound_data.y_pos, rotated_frame_x_origin, rotated_frame_y_origin, angle_difference_in_radians)
                        rotated_box_width, rotated_box_height = calculate_rotated_dimensions(current_bound_data.box_width, current_bound_data.box_height, angle_difference_in_radians)
                        #print('angle : ' + str(current_angle) + ' --> ' +str(angle))
                        bounding_box_info_to_be_written = '0 ' + str(rotated_x_pos/rotated_frame_width) + ' ' + str(rotated_y_pos/rotated_frame_height) + ' ' + str(rotated_box_width/rotated_frame_width)  + ' ' + str(rotated_box_height/rotated_frame_height)
                        #print(bounding_box_info_to_be_written)
                        new_bound_info_path = output_path + '/' + image_name + '_f{:d}_a{:d}.txt'.format(frame_number, angle)
                        new_bound_info_file = open(new_bound_info_path, "a+")
                        new_bound_info_file.write(bounding_box_info_to_be_written + '\n')
                        new_bound_info_file.close()
                
                bound_info_string = bound_info_file.readline()
                
        #new_prediction_path = output_path + '/' + image_name + 'f{:d}_a{:d}_prediction.jpg'.format(frame_number, current_angle)
        prediction_path = os.getcwd() + '/predictions.jpg'
        #shutil.copy(prediction_path, new_image_path + '_prediction.jpg')
        os.remove(prediction_path)
        os.chdir(autobound_path)

    debug_bound_info(output_path, image_name, angle_increment, frame_number)

    
     
def get_frame_with_boarders(input_frame):
    height, width, channels = input_frame.shape
    diagonal_length = int(math.hypot(height, width))
    horizontal_boarder = (diagonal_length - width)/2
    vertical_boarder = (diagonal_length - height)/2
    color = [110,100,100]
    top, bottom, left, right = [vertical_boarder, vertical_boarder, horizontal_boarder, horizontal_boarder]
    input_frame_with_boarders = cv2.copyMakeBorder(input_frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return input_frame_with_boarders

def produce_dataset_from_video(video_path, video_name):
    
    count = 0
    video_capture = cv2.VideoCapture(video_path)

    while video_capture.isOpened():
        frame_was_captured, current_frame_capture = video_capture.read()
        if frame_was_captured:
            current_frame_with_boarders = get_frame_with_boarders(current_frame_capture)
            run_detection_on_frame(current_frame_with_boarders, video_name, count)
            count += 30 
            video_capture.set(1, count)
        else:
            video_capture.release()
            break


input_path = 'Input'
image = cv2.imread('Mall_Kiosk_f360_a135.jpg', cv2.IMREAD_COLOR)
run_detection_on_frame(image, 'Debug', 0)
#for file_in_input_path in os.listdir(input_path):
#   if file_in_input_path.endswith('.mp4'):
 #      video_path = input_path + '/' + file_in_input_path
  #     video_name = os.path.splitext(file_in_input_path)[0]
   #    produce_dataset_from_video(video_path, video_name)

print('Done')
