import pandas as pd
import os
from ultralytics import YOLO
import sys 
from yolo_utils import YOLOPose
import cv2
yolo_obj = YOLOPose()



def detect(input_image):
    yolo_obj.detect(input_image, True)
    results = yolo_obj.detect(input_image, False)    
    return results


#function to
def get_folders_in_directory(directory_path):
    folders = [folder for folder in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, folder))]
    return folders


# define the directory
directory = 'split_data/train'

#get all the folder names
folders = get_folders_in_directory(directory)

rows = []


for index, folder in enumerate(folders):
    folder_path = f'{directory}/{folder}'
    
    image_names = os.listdir(folder_path)
    
    for image_name in image_names:
        
        
        image_path = f'{folder_path}/{image_name}'
        print(image_path)
        try:
            image = cv2.imread(image_path)
            results = detect(image)
            
            results = results.reshape(17,3)
            
            row = []
            for result in results:
                x = int(result[0])
                y = int(result[1])
                score = result[2]
                
                row.append(x)
                row.append(y)
                row.append(score)
                
            
            row.append(index)
            rows.append(row)
            
                
            df = pd.DataFrame(data = rows, index = None)
        except:
            print('Error occured', image_path)


df.to_csv('train_2.csv', header=None)