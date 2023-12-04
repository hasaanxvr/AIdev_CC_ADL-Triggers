import csv
import cv2
import itertools
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix





#@title Functions to visualize the pose estimation results.

def draw_prediction_on_image(
    image, person, crop_region=None, close_figure=True,
    keep_input_size=False):
  """Draws the keypoint predictions on image.
 
  Args:
    image: An numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    person: A person entity returned from the MoveNet.SinglePose model.
    close_figure: Whether to close the plt figure after the function returns.
    keep_input_size: Whether to keep the size of the input image.
 
  Returns:
    An numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  # Draw the detection result on top of the image.
  image_np = utils.visualize(image, [person])
  
  # Plot the image with detection results.
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  im = ax.imshow(image_np)
 
  if close_figure:
    plt.close(fig)
 
  if not keep_input_size:
    image_np = utils.keep_aspect_ratio_resizer(image_np, (512, 512))

  return image_np









import utils
from data import BodyPart
from ml import Movenet

#model = tf.saved_model.load('D:\MoveNet Classification\Working Directory\saved_model')
#movenet = model.signatures['serving_default']
movenet = Movenet('movenet_thunder')

# Define function to run pose estimation using MoveNet Thunder.
# You'll apply MoveNet's cropping algorithm and run inference multiple times on
# the input image to improve pose estimation accuracy.
def detect(input_tensor, inference_count=3):
  """Runs detection on an input image.
 
  Args:
    input_tensor: A [height, width, 3] Tensor of type tf.float32.
      Note that height and width can be anything since the image will be
      immediately resized according to the needs of the model within this
      function.
    inference_count: Number of times the model should run repeatly on the
      same input image to improve detection accuracy.
 
  Returns:
    A Person entity detected by the MoveNet.SinglePose.
  """
  image_height, image_width, channel = input_tensor.shape
 
  # Detect pose using the full input image
  movenet.detect(input_tensor.numpy(), reset_crop_region=True)
 
  # Repeatedly using previous detection result to identify the region of
  # interest and only croping that region to improve detection accuracy
  for _ in range(inference_count - 1):
    person = movenet.detect(input_tensor.numpy(), 
                            reset_crop_region=False)

  return person





def get_center_point(landmarks, left_bodypart, right_bodypart):
  """Calculates the center point of the two given landmarks."""

  left = tf.gather(landmarks, left_bodypart.value, axis=1)
  right = tf.gather(landmarks, right_bodypart.value, axis=1)
  center = left * 0.5 + right * 0.5
  return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
  """Calculates pose size.

  It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
  """
  # Hips center
  hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                 BodyPart.RIGHT_HIP)

  # Shoulders center
  shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER,
                                      BodyPart.RIGHT_SHOULDER)

  # Torso size as the minimum body size
  torso_size = tf.linalg.norm(shoulders_center - hips_center)

  # Pose center
  pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                     BodyPart.RIGHT_HIP)
  pose_center_new = tf.expand_dims(pose_center_new, axis=1)
  # Broadcast the pose center to the same size as the landmark vector to
  # perform substraction
  pose_center_new = tf.broadcast_to(pose_center_new,
                                    [tf.size(landmarks) // (17*2), 17, 2])

  # Dist to pose center
  d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                name="dist_to_pose_center")
  # Max dist to pose center
  max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

  # Normalize scale
  pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

  return pose_size


def normalize_pose_landmarks(landmarks):
  """Normalizes the landmarks translation by moving the pose center to (0,0) and
  scaling it to a constant pose size.
  """
  # Move landmarks so that the pose center becomes (0,0)
  pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, 
                                 BodyPart.RIGHT_HIP)
  pose_center = tf.expand_dims(pose_center, axis=1)
  # Broadcast the pose center to the same size as the landmark vector to perform
  # substraction
  pose_center = tf.broadcast_to(pose_center, 
                                [tf.size(landmarks) // (17*2), 17, 2])
  landmarks = landmarks - pose_center

  # Scale the landmarks to a constant pose size
  pose_size = get_pose_size(landmarks)
  landmarks /= pose_size

  return landmarks


def landmarks_to_embedding(landmarks_and_scores):
  """Converts the input landmarks into a pose embedding."""
  # Reshape the flat input into a matrix with shape=(17, 3)
  reshaped_inputs = keras.layers.Reshape((17, 3))(landmarks_and_scores)

  # Normalize landmarks 2D
  landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])

  # Flatten the normalized landmark coordinates into a vector
  embedding = keras.layers.Flatten()(landmarks)

  return embedding




class_names = ['lie', 'lie_to_sit', 'sit', 'sit_to_stand', 'stand']

inputs = tf.keras.Input(shape=(51))
embedding = landmarks_to_embedding(inputs)

layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.5)(layer)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

model = keras.Model(inputs, outputs)

#load the weights
model.load_weights('best_5_classes.hdf5')



import tempdir
import os


#for writing on the image
text = 'Hello, OpenCV!'
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (0, 0, 255)  # BGR format (green in this case)
text_position = (50, 50)






import imageio


if __name__ == "__main__":
    
    video_path = 'D:\\adl_module\\testing_videos\\sit_to_lie.mp4'
    
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
        
        image = tf.io.read_file(frame_path)
        image = tf.io.decode_jpeg(image)
        
        
        person = detect(image)
        detection_threshold = 0.1
  
        min_landmark_score = min([keypoint.score for keypoint in person.keypoints])
        should_keep_image = min_landmark_score >= detection_threshold
        
        landmark_detected = False
        if not should_keep_image:
            print('Not all landmarks were detected')
            landmark_detected = False

      
        pose_landmarks = np.array(
              [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                for keypoint in person.keypoints],
              dtype=np.float32)

          # Write the landmark coordinates to its per-class CSV file
        coordinates = pose_landmarks.flatten().astype(float).tolist()

        
        
          

        input_data = np.array(coordinates).reshape(1, 51)
        output = model(input_data)
        index = tf.argmax(output[0])
        
        
        score = output[0][index]
          
        prediction = class_names[index]
        
        cv2.putText(frame, f'{prediction}, score: {score}', text_position, font, font_scale, text_color, font_thickness)

        cv2.imshow('Video', frame)
        
        output_video.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        i+=1

    # Exit the loop if 'q' key is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
          
          break
    
    
    
    # Release the resources
    cap.release()
    output_video.close()
    cv2.destroyAllWindows()

    print(f"Video saved as {output_file}")
    
    try:
        os.rmdir(temp_dir)
        print(f'Temporary directory {temp_dir} deleted.')
    except OSError as e:
        print(f'Error deleting temporary directory: {e}')
    
    
    
    
    """
  image = tf.io.read_file('data\sit\jehanzeb_sittl_slow2_903.25.jpg')
  image = tf.io.decode_jpeg(image)
  person = detect(image)
  
  #img = draw_prediction_on_image(image.numpy(), person, crop_region=None, 
   #                            close_figure=False, keep_input_size=True)
  
  #cv2.imshow('test_images', img)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()
 
  detection_threshold = 0.1
  
  min_landmark_score = min([keypoint.score for keypoint in person.keypoints])
  should_keep_image = min_landmark_score >= detection_threshold
  if not should_keep_image:
    print('Not all landmarks were detected')
    
  pose_landmarks = np.array(
              [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                for keypoint in person.keypoints],
              dtype=np.float32)

          # Write the landmark coordinates to its per-class CSV file
  coordinates = pose_landmarks.flatten().astype(float).tolist()
  

  input_data = np.array(coordinates).reshape(1, 51)
  output = model(input_data)
  print(output)
  """