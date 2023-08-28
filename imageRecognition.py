#import cv2
#from PIL import Image
#import os
from tensorflow import keras
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
val_size = int(len(pipeline)*.3)
test_size = int(len(pipeline)*.1)
print(train_size)
print(val_size)
print(test_size)
print(len(pipeline))
