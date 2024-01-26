from PIL import Image
import cv2
import numpy as np
from pathlib import Path

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
        
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def risky_standing(right_hip, left_hip,right_knee,left_knee):
        if((right_hip[0] < right_knee[0]) or (left_hip[0] < left_knee[0])):
            Th = True
        else:
            Th = False

        return Th

def get_hip_flexion_angle(shoulder, knee, hip):
        
        angle_hip = calculate_angle(shoulder, hip, knee)
        hip_angle = 180 - angle_hip
        return hip_angle


def risky_sitting(right_shoulder, left_shoulder, right_knee, left_knee, right_hip, left_hip):
    if((right_shoulder<right_knee) or (left_shoulder<left_knee)):
            Th1 = True
    else: Th1 = False
    if((right_hip < right_knee) or (left_hip < left_knee)):
            Th2 = True
    else:
            Th2 = False
    return Th1, Th2


