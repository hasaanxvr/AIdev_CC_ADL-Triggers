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

model_cls = keras.Model(inputs, outputs)

#load the weights
model_cls.load_weights('best_5_classes.hdf5')


# Download the model from TF Hub.
model = tf.saved_model.load('saved_model_singlepose')
movenet = model.signatures['serving_default']

# Run model inference

#for writing on the image
text = 'Hello, OpenCV!'
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (0, 0, 255)  # BGR format (green in this case)
text_position = (50, 50)







def init_crop_region(image_height, image_width):
    """Defines the default crop region.

    The function provides the initial crop region (pads the full image from
    both sides to make it a square image) when the algorithm cannot reliably
    determine the crop region from the previous frame.

    Args:
      image_height (int): The input image width
      image_width (int): The input image height

    Returns:
      crop_region (dict): The default crop region.
    """
    if image_width > image_height:
      x_min = 0.0
      box_width = 1.0
      # Pad the vertical dimension to become a square image.
      y_min = (image_height / 2 - image_width / 2) / image_height
      box_height = image_width / image_height
    else:
      y_min = 0.0
      box_height = 1.0
      # Pad the horizontal dimension to become a square image.
      x_min = (image_width / 2 - image_height / 2) / image_width
      box_width = image_height / image_width
      

    return {
        'y_min': y_min,
        'x_min': x_min,
        'y_max': y_min + box_height,
        'x_max': x_min + box_width,
        'height': box_height,
        'width': box_width
    }




from data import person_from_keypoints_with_scores

import imageio

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
        
        print(keypoints_with_scores)
        for idx in range(len(BodyPart)):
            keypoints_with_scores[idx, 0] = crop_region[
          'y_min'] + crop_region['height'] * keypoints_with_scores[idx, 0]
            keypoints_with_scores[idx, 1] = crop_region[
          'x_min'] + crop_region['width'] * keypoints_with_scores[idx, 1]
            
        
        
        #input_data = np.array(keypoints_with_scores).reshape(1, 51)
        
        person = person_from_keypoints_with_scores(keypoints_with_scores, image_height, image_width)
        
        pose_landmarks = np.array(
              [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                for keypoint in person.keypoints],
              dtype=np.float32)
        
        coordinates = pose_landmarks.flatten().astype(float).tolist()
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









