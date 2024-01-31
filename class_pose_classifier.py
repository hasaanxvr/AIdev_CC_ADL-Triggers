"""This file defines the class for the pose classifier model that is used to classify poses based on landmarks"""


import tensorflow as tf
from tensorflow import keras
from utils import landmarks_to_embedding

class PoseClassifier():
    def __init__(self):
        
        self.class_names = ['lie', 'lie to sit', 'sit', 'sit to stand', 'stand']
        
        #define the architecture of the model
        print('Building the pose classifier model')
        inputs = tf.keras.Input(shape=(51))
        embedding = landmarks_to_embedding(inputs)
        layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
        layer = keras.layers.Dropout(0.5)(layer)
        layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
        layer = keras.layers.Dropout(0.5)(layer)
        outputs = keras.layers.Dense(len(self.class_names), activation="softmax")(layer)
        self.model_cls = keras.Model(inputs, outputs)
        
        print('loading the weights of the pose classifier')
        self.model_cls.load_weights('weights.best_1.hdf5')
        
        
    def detect(self, landmarks):
        return self.model_cls(landmarks)
        
