import cv2
import os
import shutil
from os import path
import numpy as np
import imutils
import math 

output_path = 'Output'
angle_increment = 15
for output_file in os.listdir(output_path):
    current_boxes_path = output_path + '/' + output_file + '/Boxes'
    current_dataset_path = output_path + '/' + output_file 
    list_of_existing_image_paths = set([])
    for boxes_file in sorted(os.listdir(current_boxes_path)):
        name_length = len(output_file)
        if boxes_file.endswith('.jpg') or boxes_file.endswith('.txt'):
            existing_image_path = output_file + '_f' + boxes_file[name_length+2:-10].split('_')[0]
            list_of_existing_image_paths.add(existing_image_path)
            
    if boxes_file.endswith('.jpg') or boxes_file.endswith('.txt'):
        print output_file
    
    box_count = 0
    for existing_image_path in list_of_existing_image_paths:
        box_count += 1
        current_box_path = current_boxes_path + '/' + existing_image_path + '_a0_boxes.jpg'
        current_boxes_image = cv2.imread(current_boxes_path + '/' + existing_image_path + '_a0_boxes.jpg')
        cv2.imshow(current_box_path + ' ' + str(box_count) + '/' + str(len(list_of_existing_image_paths)) , current_boxes_image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
        response = raw_input("Delete this? : ") 
        print response
        if response is 'y':
            print '     Deleting : ' + existing_image_path
            for angle in np.arange(0, 360, angle_increment):
                path_of_boxes_to_delete = current_boxes_path + '/' + existing_image_path + '_a{:d}.jpg'.format(angle)
                path_of_textfile_to_delete = current_dataset_path + '/' + existing_image_path + '_a{:d}.txt'.format(angle)
                os.remove(path_of_textfile_to_delete)
                path_of_image_to_delete =  current_dataset_path + '/' + existing_image_path + '_a{:d}.jpg'.format(angle)
                os.remove(path_of_image_to_delete)
    #print list_of_existing_image_paths
print('Done with all videos.')
