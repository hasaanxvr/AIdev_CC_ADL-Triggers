# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import numpy as np
from movenet_utils import Movenet


import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm
import time

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data import BodyPart
from utils import landmarks_to_embedding
from utils import load_class_names
from utils import core_points_detected
from keras.models import load_model

from data import person_from_keypoints_with_scores
import imageio

#initilize the movenet model
movenet = Movenet()


def detect(input):
  input = input.numpy()
  movenet.detect(input, True)
  return movenet.detect(input, False)



class_names = load_class_names()



inputs = tf.keras.Input(shape=(51))
embedding = landmarks_to_embedding(inputs)

layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.5)(layer)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

model_cls = keras.Model(inputs, outputs)

print('Loading the pose classifier model...')
#model_cls = load_model('pose_classifier.h5')

print('Loading the model weights')
model_cls.load_weights('best_5_classes.hdf5')



# Download the model from TF Hub.
#model = tf.saved_model.load('saved_model_singlepose')
#movenet = model.signatures['serving_default']

#for writing on the image
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (0, 0, 255)  # BGR format (green in this case)
text_position = (50, 50)





def get_points(coordinates):
    points = []
    
    i = 0
    
    while i < len(coordinates):
      point = (coordinates[i], coordinates[i+1])
      points.append(point)
      
      i+=3
      
    return points









def add_state(current_state, states):
  if current_state == -1:
    return states  
  
  if len(states) > 0:
    previous_state = states[len(states) - 1]
    
    if previous_state != current_state:
      states.append(current_state)
    
    return states
  
  states.append(current_state)
  return states



def create_transition_embedding(states):
  embedding = ''
  for state in states:
    
    embedding += str(state)

  return embedding





previous_state = -1
current_state = -1

transition_state = -1


from transitions import transition_valid

if __name__ == "__main__":
    
    
    
    
    #create a temporary directory
    
    temp_dir = tempfile.mkdtemp()
    print(f'Temporary directory: {temp_dir}')
    

    video_name = 'adlvideo33.mp4'

    cap = cv2.VideoCapture(f'videos/{video_name}')
    #cap = cv2.VideoCapture()

    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Specify the output video file path and codec
    output_video_path = f'results/{video_name}'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out = cv2.VideoWriter(output_video_path, fourcc, 5 , (width, height))
    
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    i = 0
    while True:
      
        ret, frame = cap.read()
          
        frame_path = f'{temp_dir}/frame_{i}.jpg'
        cv2.imwrite(frame_path, frame)
        
        image_path = frame_path
        image = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(image)
      

        
        #perform the detection
        person = detect(image)       
        
        if core_points_detected(person.keypoints):
          
          pose_landmarks = np.array(
                [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                  for keypoint in person.keypoints],
                dtype=np.float32)
          
          coordinates = pose_landmarks.flatten().astype(float).tolist()
          
          
          
          input_data = np.array(coordinates).reshape(1, 51)
      
          
          output = model_cls(input_data)
          
          index = tf.argmax(output[0])
          
          previous_state = current_state
          current_state = index
          
          
          score = output[0][index]
          prediction = class_names[index]
          
          if index == 1:
            if output[0][0] > 0.1:
              prediction = 'lie'
              current_state = 0
          
          """
          if (current_state == 5 and previous_state == 4) or (current_state == 6 and previous_state == 4):
            cv2.putText(frame, f'The patient is trying to stand up!', (100,500), font, font_scale, text_color, font_thickness)
          
          if current_state == 2 and previous_state == 1:
            cv2.putText(frame, f'The patient is trying to sit!', (100,500), font, font_scale, text_color, font_thickness)
          """
          
          
          if (current_state == 3 and previous_state == 2) or (current_state == 4 and previous_state == 2):
            cv2.putText(frame, f'The patient is trying to stand up!', (100,500), font, font_scale, text_color, font_thickness)
            transition_state = 1
            
          if current_state == 1 and previous_state == 0:
            cv2.putText(frame, f'The patient is trying to sit!', (100,500), font, font_scale, text_color, font_thickness)
            
            
          if transition_state == 1:
            if current_state == 2:
              cv2.putText(frame, f'failed sit to stand transition!', (100,500), font, font_scale, text_color, font_thickness)
              transition_state = -1
            if current_state == 4:
              cv2.putText(frame, f'sit to stand transition complete', (200,500), font, font_scale, (0, 255, 0), font_thickness)
              transition_state = -1
            
          points = get_points(coordinates)
          
          #draw landmarks on the image
          for landmark in points:
            x, y = landmark
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Circle radius: 5, color: (0, 255, 0)

          
          score = output[0][index]
          prediction = class_names[index]
          
          if index == 1:
              lie_score = output[0][0]
              if lie_score > 0.01:
                prediction = 'lie'
          
        
          cv2.putText(frame, f'{prediction}', text_position, font, font_scale, text_color, font_thickness)
          cv2.putText(frame, f'{output[0][0]}, {output[0][1]}', (50, 200), font, font_scale, text_color, font_thickness)
          

        cv2.imshow('Video', frame)
        out.write(frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
          
          break

        
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    
    
    try:
        import shutil
        shutil.rmtree(temp_dir)

            #os.rmdir(temp_dir)
        print(f'Temporary directory {temp_dir} deleted.')
    except OSError as e:
            print(f'Error deleting temporary directory: {e}')









