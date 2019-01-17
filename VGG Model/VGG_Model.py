# -*- coding: utf-8 -*-

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16

#load model
model = VGG16()

#load an image from file
image = load_img('mug.jpg', target_size =(224,224))

#convert the image pixels to a numpy array
image = img_to_array(image)

#reshape data for the model
image = image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))

#prepare the image for the VGG model
image = preprocess_input(image)

#predict the probability across all output classes
yhat = model.predict(image)

#convert the probability to class labels
label = decode_predictions(yhat)

#retrieve the most likely result, e.g. hight probability
label = label[0][0]
#print the classification
print('%s(%.2f%%)'%(label[1],label[2]*100))
