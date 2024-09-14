import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels)=data.load_data()
class_names=['t-shirt/top','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot' ]
train_images=train_images/255.0
test_images=test_images/255.0
print(train_images[7])
#to display the image
plt.imshow(train_images[7],cmap=plt.cm.binary)
plt.show()


"""model =keras.sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.dense(128,activation="relu"),
    keras.layers.dense(10,activation="softmax")
])"""
"""What is the Softmax Function?
The softmax function is like a mathematical "squeezer" 
that takes a bunch of numbers (called logits) and turns them into 
probabilities that add up to 1. This is really useful when you want to decide which 
category something belongs to."""