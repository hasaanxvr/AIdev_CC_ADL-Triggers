from ultralytics import YOLO
import cv2
from utils import landmarks_to_embedding
import tensorflow as tf
from tensorflow import keras
from yolo_utils import YOLOPose
from class_pose_classifier import PoseClassifier
from risky_behaviors import risky_hip_flexion_angle, risky_sitting
from class_body_part import BodyPart

def detect(input_image):
    
    yolo_obj.detect(input_image, True)
    results = yolo_obj.detect(input_image, False)    
    return results



def get_coordinates(bodyPart, landmarks):
    x = landmarks[bodyPart][0]
    y = landmarks[bodyPart][1]
    
    return (x,y)



yolo_obj = YOLOPose()
pose_classifier = PoseClassifier()


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (0, 0, 255)  # BGR format (green in this case)
text_position = (50, 50)



# Path to the video file
video_path = 'adlvideo26.mp4'

# Open the video file
cap = cv2.VideoCapture(f'videos/{video_path}')
cap = cv2.VideoCapture(0)


previous_state = -1
current_state = -1

# Read and display frames from the video
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("End of video.")
        break

    # Display the frame
    landmarks = detect(frame)

    
    right_shoulder = get_coordinates(BodyPart.RIGHT_SHOULDER.value, landmarks)
    left_shoulder = get_coordinates(BodyPart.LEFT_SHOULDER.value, landmarks)
    right_knee = get_coordinates(BodyPart.RIGHT_KNEE.value, landmarks)
    left_knee = get_coordinates(BodyPart.LEFT_KNEE.value, landmarks)
    right_hip = get_coordinates(BodyPart.RIGHT_HIP.value, landmarks)
    left_hip = get_coordinates(BodyPart.LEFT_HIP.value, landmarks) 
    
    
    
    risky_knee_distance = risky_sitting(right_shoulder, left_shoulder, right_knee, left_knee)
    
    risky_hip_flexion = risky_hip_flexion_angle(right_shoulder, right_hip, right_knee, left_shoulder, left_hip, left_knee)
        
    landmarks = landmarks.reshape(1, 51)
    
    
    
    i = 0
    while i < 51:
        x = landmarks[0][i]
        y = landmarks[0][i+1]
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Circle radius: 5, color: (0, 255, 0)
        i+= 3
            
        
    output = pose_classifier.detect(landmarks)
    
    index = tf.argmax(output[0])      
    previous_state = current_state
    current_state = index
    
    score = output[0][index]
    prediction = pose_classifier.class_names[index]
    
    
    if (current_state == 3 and previous_state == 2) or (current_state == 4 and previous_state == 2):
        cv2.putText(frame, f'The patient is trying to stand up!', (100,500), font, font_scale, text_color, font_thickness)
          
    #cv2.putText(frame, f'{prediction}, {score}', text_position, font, font_scale, text_color, font_thickness)
    #cv2.putText(frame, f'{prediction}, {output[0]}', (100, 300), font, font_scale, (0, 255, 0), font_thickness)
    
    cv2.putText(frame, f'{risky_hip_flexion}',(100, 300), font, font_scale, text_color, font_thickness)
    cv2.imshow('Video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
