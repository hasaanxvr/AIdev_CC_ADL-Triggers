
from ultralytics import YOLO
from typing import Dict, List
import numpy as np
from data import BodyPart
import cv2
import tensorflow as tf

class YOLOPose(object):
    
    _MIN_CROP_KEYPOINT_SCORE = 0.2
    _TORSO_EXPANSION_RATIO = 1.9
    _BODY_EXPANSION_RATIO = 1.2
    
    def __init__(self):
        self.yolo_pose = YOLO("yolov8m-pose.pt")
        
        self._input_height = 640
        self._input_width = 640
    
    
    def init_crop_region(self, image_height: int,
                       image_width: int) -> Dict[(str, float)]:
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
        
    def _torso_visible(self, keypoints: np.ndarray) -> bool:
        """Checks whether there are enough torso keypoints.

        This function checks whether the model is confident at predicting one of
        the shoulders/hips which is required to determine a good crop region.

        Args:
        keypoints: Detection result of Movenet model.

        Returns:
        True/False
        """
        left_hip_score = keypoints[BodyPart.LEFT_HIP.value, 2]
        right_hip_score = keypoints[BodyPart.RIGHT_HIP.value, 2]
        left_shoulder_score = keypoints[BodyPart.LEFT_SHOULDER.value, 2]
        right_shoulder_score = keypoints[BodyPart.RIGHT_SHOULDER.value, 2]

        left_hip_visible = left_hip_score > YOLOPose._MIN_CROP_KEYPOINT_SCORE
        right_hip_visible = right_hip_score > YOLOPose._MIN_CROP_KEYPOINT_SCORE
        left_shoulder_visible = left_shoulder_score > YOLOPose._MIN_CROP_KEYPOINT_SCORE
        right_shoulder_visible = right_shoulder_score > YOLOPose._MIN_CROP_KEYPOINT_SCORE

        return ((left_hip_visible or right_hip_visible) and
                (left_shoulder_visible or right_shoulder_visible))
        
    
    def _determine_torso_and_body_range(self, keypoints: np.ndarray,
                                      target_keypoints: Dict[(str, float)],
                                      center_y: float,
                                      center_x: float) -> List[float]:
        """Calculates the maximum distance from each keypoints to the center.

        The function returns the maximum distances from the two sets of keypoints:
        full 17 keypoints and 4 torso keypoints. The returned information will
        be used to determine the crop size. See determine_crop_region for more
        details.

        Args:
        keypoints: Detection result of Movenet model.
        target_keypoints: The 4 torso keypoints.
        center_y (float): Vertical coordinate of the body center.
        center_x (float): Horizontal coordinate of the body center.

        Returns:
        The maximum distance from each keypoints to the center location.
        """
        torso_joints = [
            BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER, BodyPart.LEFT_HIP,
            BodyPart.RIGHT_HIP
        ]
        max_torso_yrange = 0.0
        max_torso_xrange = 0.0
        for joint in torso_joints:
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_torso_yrange:
                max_torso_yrange = dist_y
            if dist_x > max_torso_xrange:
                max_torso_xrange = dist_x

        max_body_yrange = 0.0
        max_body_xrange = 0.0
        for idx in range(len(BodyPart)):
            if keypoints[BodyPart(idx).value, 2] < YOLOPose._MIN_CROP_KEYPOINT_SCORE:
                continue
            dist_y = abs(center_y - target_keypoints[joint][0])
            dist_x = abs(center_x - target_keypoints[joint][1])
            if dist_y > max_body_yrange:
                max_body_yrange = dist_y

            if dist_x > max_body_xrange:
                max_body_xrange = dist_x

        return [
            max_torso_yrange, max_torso_xrange, max_body_yrange, max_body_xrange
        ]


    def _determine_crop_region(self, keypoints: np.ndarray, image_height: int,
                                image_width: int) -> Dict[(str, float)]:
        """Determines the region to crop the image for the model to run inference on.

        The algorithm uses the detected joints from the previous frame to
        estimate the square region that encloses the full body of the target
        person and centers at the midpoint of two hip joints. The crop size is
        determined by the distances between each joints and the center point.
        When the model is not confident with the four torso joint predictions,
        the function returns a default crop which is the full image padded to
        square.

        Args:
        keypoints: Detection result of Movenet model.
        image_height (int): The input image width
        image_width (int): The input image height

        Returns:
        crop_region (dict): The crop region to run inference on.
        """
        
        # Convert keypoint index to human-readable names.
        target_keypoints = {}
        for idx in range(len(BodyPart)):
            target_keypoints[BodyPart(idx)] = [
                keypoints[idx, 0] * image_height, keypoints[idx, 1] * image_width
            ]

        # Calculate crop region if the torso is visible.
        if self._torso_visible(keypoints):
            center_y = (target_keypoints[BodyPart.LEFT_HIP][0] +
                        target_keypoints[BodyPart.RIGHT_HIP][0]) / 2
            center_x = (target_keypoints[BodyPart.LEFT_HIP][1] +
                        target_keypoints[BodyPart.RIGHT_HIP][1]) / 2

            (max_torso_yrange, max_torso_xrange, max_body_yrange,
            max_body_xrange) = self._determine_torso_and_body_range(
                keypoints, target_keypoints, center_y, center_x)

            crop_length_half = np.amax([
                max_torso_xrange * YOLOPose._TORSO_EXPANSION_RATIO,
                max_torso_yrange * YOLOPose._TORSO_EXPANSION_RATIO,
                max_body_yrange * YOLOPose._BODY_EXPANSION_RATIO,
                max_body_xrange * YOLOPose._BODY_EXPANSION_RATIO
            ])

            # Adjust crop length so that it is still within the image border
            distances_to_border = np.array(
                [center_x, image_width - center_x, center_y, image_height - center_y])
            crop_length_half = np.amin(
                [crop_length_half, np.amax(distances_to_border)])

            # If the body is large enough, there's no need to apply cropping logic.
            if crop_length_half > max(image_width, image_height) / 2:
                return self.init_crop_region(image_height, image_width)
            # Calculate the crop region that nicely covers the full body.
            else:
                crop_length = crop_length_half * 2
            crop_corner = [center_y - crop_length_half, center_x - crop_length_half]
        
        
            return {
                'y_min':
                    crop_corner[0] / image_height,
                'x_min':
                    crop_corner[1] / image_width,
                'y_max': (crop_corner[0] + crop_length) / image_height,
                'x_max': (crop_corner[1] + crop_length) / image_width,
                'height': (crop_corner[0] + crop_length) / image_height -
                            crop_corner[0] / image_height,
                'width': (crop_corner[1] + crop_length) / image_width -
                        crop_corner[1] / image_width
            }
    
        else:
            return self.init_crop_region(image_height, image_width)



    def _crop_and_resize(
        self, image: np.ndarray, crop_region: Dict[(str, float)],
        crop_size: (int, int)) -> np.ndarray:
        """Crops and resize the image to prepare for the model input."""
        y_min, x_min, y_max, x_max = [
            crop_region['y_min'], crop_region['x_min'], crop_region['y_max'],
            crop_region['x_max']
        ]

        crop_top = int(0 if y_min < 0 else y_min * image.shape[0])
        crop_bottom = int(image.shape[0] if y_max >= 1 else y_max * image.shape[0])
        crop_left = int(0 if x_min < 0 else x_min * image.shape[1])
        crop_right = int(image.shape[1] if x_max >= 1 else x_max * image.shape[1])

        padding_top = int(0 - y_min * image.shape[0] if y_min < 0 else 0)
        padding_bottom = int((y_max - 1) * image.shape[0] if y_max >= 1 else 0)
        padding_left = int(0 - x_min * image.shape[1] if x_min < 0 else 0)
        padding_right = int((x_max - 1) * image.shape[1] if x_max >= 1 else 0)

        
        
        output_image = image[crop_top:crop_bottom, crop_left:crop_right]
        output_image = cv2.copyMakeBorder(output_image, padding_top, padding_bottom,
                                        padding_left, padding_right,
                                        cv2.BORDER_CONSTANT)
        output_image = cv2.resize(output_image, (crop_size[0], crop_size[1]))

        return output_image


    def detect(self, image, reset_crop_region):
      
        image_height, image_width = image.shape[0], image.shape[1]
        
        
        #image = tf.cast(tf.image.resize_with_pad(image, image_height, image_width), dtype=tf.int32)
    
        if reset_crop_region:
        # Set crop region for the first frame.
            self._crop_region = self.init_crop_region(image_height, image_width)


        image = self._crop_and_resize(image, self._crop_region, (self._input_height, self._input_width))
        
        outputs = self.yolo_pose(image)
                
        keypoints_with_scores = outputs[0][0].keypoints.data
        
        keypoints_with_scores = keypoints_with_scores.numpy()
        
        keypoints_with_scores = keypoints_with_scores.reshape(17,3)
        
        
        self._crop_region = self._determine_crop_region(keypoints_with_scores, image_height, image_width)
        return keypoints_with_scores
        
        #self._crop_region = self._determine_crop_region(keypoints_with_scores, image_height, image_width)  
        #return keypoints_with_scores

