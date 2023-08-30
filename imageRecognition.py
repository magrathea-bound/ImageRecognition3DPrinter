#import cv2
#from PIL import Image
#import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt

imageFolder = 'images'


"""
************************************************************************************
Data Clean
************************************************************************************
Data was clean so only a basic check was performed.
A try, except clause could be implemented around cv2.imread in the case that images are corrupt.
"""
"""
photoType = [0, 0]

for status in os.listdir(imageFolder):
    for photo in os.listdir(os.path.join(imageFolder, status)):
        img = cv2.imread(os.path.join(imageFolder, status, photo))
        imgDet = Image.open(os.path.join(imageFolder, status, photo))
        if imgDet.format != 'JPEG':
            os.remove(imageFolder + photo)
        if photo[:7] == 'scratch':
            photoType[0] = photoType[0] + 1
        else:
            photoType[1] = photoType[1] + 1

print("Good Print Counts: " + str(photoType[0]))
print("Bad Print Counts: " + str(photoType[1]))
print("Total Picture count: " + str(photoType[0] + photoType[1]))
"""

"""
************************************************************************************
Creation of pipeline
************************************************************************************
"""
#Set up Data Pipeline
pipeline = keras.utils.image_dataset_from_directory('images', class_names = ['good','fail'])
data_iterator = pipeline.as_numpy_iterator()
batch = data_iterator.next()
print(batch[1])

"""
#Verify image label
#class_names argument assigned 0 to good images and 1 to fail images
fig, ax = plt.subplots(ncols=4, figsize=(20,20)) 

for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()
"""


"""
************************************************************************************
Pre-Processing
************************************************************************************
"""
#Scaling Data
pipeline = pipeline.map(lambda x,y: (x/255, y))
pipeline.as_numpy_iterator().next()

#Splitting Data
train_size = int(len(pipeline) * .6)
val_size = int(len(pipeline)*.3) + 1
test_size = int(len(pipeline)*.1) + 1

train = pipeline.take(train_size)
val = pipeline.skip(train_size).take(val_size)
test = pipeline.skip(train_size + val_size).take(test_size)

"""
************************************************************************************
Model Construction
************************************************************************************
"""

model = Sequential()

model.add(Conv2D(16, (3,3), 1, activation = 'relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile('adam', loss = tf.losses.BinaryCrossentropy(), metrics = ['accuracy'])

model.summary()

"""
************************************************************************************
Model Training
************************************************************************************
"""

logdir='logs'  
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)  
hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback]) 

plt.plot(hist.history['loss'], color='teal', label='loss') 
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')  
plt.legend(loc="upper left") 
plt.savefig('Loss')

plt.plot(hist.history['accuracy'], color='teal', label='loss') 
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')  
plt.legend(loc="upper left") 
plt.savefig('Accuracy')
"""
************************************************************************************
Evaluate
************************************************************************************
"""

from tensorflow.keras.metrics import BinaryAccuracy
accuracy = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yPred = model.predict(X)
    accuracy.update_state(y, yPred)

print(accuracy.result().numpy())


