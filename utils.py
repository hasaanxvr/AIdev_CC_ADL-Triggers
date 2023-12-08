import tensorflow as tf
import utils
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from tensorflow import keras
from data import BodyPart

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
    
    
    

def load_class_names():
  file_path = 'class_names.txt'

  with open(file_path, 'r') as file:
      class_names = [class_name.strip() for class_name in file.readlines()]

  return class_names


import numpy as np
from typing import List, Tuple
from data import Person
import cv2

def visualize(
    image: np.ndarray,
    list_persons: List[Person],
    keypoint_color: Tuple[int, ...] = None,
    keypoint_threshold: float = 0.05,
    instance_threshold: float = 0.1,
) -> np.ndarray:
  """Draws landmarks and edges on the input image and return it.

  Args:
    image: The input RGB image.
    list_persons: The list of all "Person" entities to be visualize.
    keypoint_color: the colors in which the landmarks should be plotted.
    keypoint_threshold: minimum confidence score for a keypoint to be drawn.
    instance_threshold: minimum confidence score for a person to be drawn.

  Returns:
    Image with keypoints and edges.
  """
  for person in list_persons:
    if person.score < instance_threshold:
      continue

    keypoints = person.keypoints
    bounding_box = person.bounding_box

    # Assign a color to visualize keypoints.
    if keypoint_color is None:
      if person.id is None:
        # If there's no person id, which means no tracker is enabled, use
        # a default color.
        person_color = (0, 255, 0)
      else:
        # If there's a person id, use different color for each person.
        person_color = COLOR_LIST[person.id % len(COLOR_LIST)]
    else:
      person_color = keypoint_color

    # Draw all the landmarks
    for i in range(len(keypoints)):
      if keypoints[i].score >= keypoint_threshold:
        cv2.circle(image, keypoints[i].coordinate, 2, person_color, 4)

    # Draw all the edges
    for edge_pair, edge_color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (keypoints[edge_pair[0]].score > keypoint_threshold and
          keypoints[edge_pair[1]].score > keypoint_threshold):
        cv2.line(image, keypoints[edge_pair[0]].coordinate,
                 keypoints[edge_pair[1]].coordinate, edge_color, 2)

    # Draw bounding_box with multipose
    if bounding_box is not None:
      start_point = bounding_box.start_point
      end_point = bounding_box.end_point
      cv2.rectangle(image, start_point, end_point, person_color, 2)
      # Draw id text when tracker is enabled for MoveNet MultiPose model.
      # (id = None when using single pose model or when tracker is None)
      if person.id:
        id_text = 'id = ' + str(person.id)
        cv2.putText(image, id_text, start_point, cv2.FONT_HERSHEY_PLAIN, 1,
                    (0, 0, 255), 1)

  return image




def core_points_detected(keypoints):
  
  LEFT_SHOULDER_SCORE = keypoints[5].score
  RIGHT_SHOULDER_SCORE = keypoints[6].score
  
  LEFT_HIP_SCORE = keypoints[11].score
  RIGHT_HIP_SCORE = keypoints[12].score
  
  LEFT_KNEE_SCORE = keypoints[13].score
  RIGHT_KNEE_SCORE = keypoints[14].score
  
  
  if (LEFT_SHOULDER_SCORE < 0.1 or
    RIGHT_SHOULDER_SCORE < 0.1 or
    LEFT_HIP_SCORE < 0.1 or
    RIGHT_HIP_SCORE < 0.1 or
    LEFT_KNEE_SCORE < 0.1 or
    RIGHT_KNEE_SCORE < 0.1):
      return False
    
  return True