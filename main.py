# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import numpy as np



import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm


import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data import BodyPart
from utils import landmarks_to_embedding
from utils import init_crop_region
from utils import load_class_names
from utils import core_points_detected
from keras.models import load_model

from data import person_from_keypoints_with_scores
import imageio



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
model_cls.load_weights('weights.best.hdf5')



# Download the model from TF Hub.
model = tf.saved_model.load('saved_model_singlepose')
movenet = model.signatures['serving_default']


#for writing on the image
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (0, 0, 255)  # BGR format (green in this case)
text_position = (50, 50)






from utils import visualize

if __name__ == "__main__":
    
   
    
    #create a temporary directory
    
    temp_dir = tempfile.mkdtemp()
    print(f'Temporary directory: {temp_dir}')
    
    output_file = 'sit_stand_sofa.mp4'
    output_video = imageio.get_writer(output_file, fps=5)

    cap = cv2.VideoCapture(0)
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
        image = tf.compat.v1.image.decode_jpeg(image)
        image = tf.expand_dims(image, axis=0)
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image_height = 256
        image_width = 256
        image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
        
        
        outputs = movenet(image)
        keypoints = outputs['output_0']
        keypoints_numpy = keypoints.numpy()
        keypoints_with_scores = keypoints_numpy.reshape(17,3)
        crop_region = init_crop_region(image_height, image_width)
        
        for idx in range(len(BodyPart)):
            keypoints_with_scores[idx, 0] = crop_region[
          'y_min'] + crop_region['height'] * keypoints_with_scores[idx, 0]
            keypoints_with_scores[idx, 1] = crop_region[
          'x_min'] + crop_region['width'] * keypoints_with_scores[idx, 1]
            
        
        
        #input_data = np.array(keypoints_with_scores).reshape(1, 51)
        
        person = person_from_keypoints_with_scores(keypoints_with_scores, image_height, image_width)
        
        
        
        if core_points_detected(person.keypoints):
          pose_landmarks = np.array(
                [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                  for keypoint in person.keypoints],
                dtype=np.float32)
          
          coordinates = pose_landmarks.flatten().astype(float).tolist()
          
          
          detected_points = 0
          for i in range(0,51,3):
            #x = int(coordinates[i])  # Convert to integer
            #y = int(coordinates[i+1])
            #cv2.circle(frame, (x,y), 5, (0, 0, 255), -1)  # Draw a red filled circle
            
            score = coordinates[i+2]
            if score > 0.1:
              detected_points +=1
              
          print(detected_points)

          input_data = np.array(coordinates).reshape(1, 51)
          
          output = model_cls(input_data)
          index = tf.argmax(output[0])
          score = output[0][index]
          prediction = class_names[index]

          cv2.putText(frame, f'{prediction}, score: {score}', text_position, font, font_scale, text_color, font_thickness)

        cv2.imshow('Video', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
          
          break

        
        
    cap.release()
    cv2.destroyAllWindows()

    
    
    try:
        import shutil
        shutil.rmtree(temp_dir)

            #os.rmdir(temp_dir)
        print(f'Temporary directory {temp_dir} deleted.')
    except OSError as e:
            print(f'Error deleting temporary directory: {e}')









