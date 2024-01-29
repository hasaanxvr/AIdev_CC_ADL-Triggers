from ultralytics import YOLO
import cv2
from utils import landmarks_to_embedding
import tensorflow as tf
from tensorflow import keras
from yolo_utils import YOLOPose


yolo_obj = YOLOPose()





def detect(input_image):
    
    
    #yolo_obj.detect(input_image, True)
    
    yolo_obj.detect(input_image, True)
    
    results = yolo_obj.detect(input_image, False)    
    return results









#initialize the YOLO-Pose model

#define a list of class names
class_names = ['lie', 'lie to sit', 'sit', 'sit to stand', 'stand']

#define the pose classifier
inputs = tf.keras.Input(shape=(51))
embedding = landmarks_to_embedding(inputs)
layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
layer = keras.layers.Dropout(0.5)(layer)
outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)
model_cls = keras.Model(inputs, outputs)

print('Loading the model weights')
model_cls.load_weights('weights.best_1.hdf5')


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2
text_color = (0, 0, 255)  # BGR format (green in this case)
text_position = (50, 50)



# Path to the video file
video_path = 'adlvideo130.mp4'

# Open the video file
cap = cv2.VideoCapture(f'videos/{video_path}')


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
    
    landmarks = landmarks.reshape(1, 51)
    
    
    i = 0
    while i < 51:

        x = landmarks[0][i]
        
        y = landmarks[0][i+1]
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Circle radius: 5, color: (0, 255, 0)
        i+= 3
            
        
    output = model_cls(landmarks)

    index = tf.argmax(output[0])
          
    score = output[0][index]
    prediction = class_names[index]
          
    print(prediction, score)
    cv2.putText(frame, f'{prediction}, {score}', text_position, font, font_scale, text_color, font_thickness)
    cv2.putText(frame, f'{prediction}, {output[0]}', (100, 300), font, font_scale, (0, 255, 0), font_thickness)
    cv2.imshow('Video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
