# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 22:19:59 2019

@author: ncthu
"""

from matplotlib import pyplot as plt
import cv2

for i in range(1,5,1):
    plt.rcParams["figure.figsize"] = (15, 12)
    face_cascade = cv2.CascadeClassifier("C:\\Users\\ncthu\\Anaconda3\\Lib\\site-packages\\cv2\data\\haarcascade_frontalface_default.xml")
        
    img = cv2.imread('img'+str(i)+'.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 10)
    for (x, y, w, h) in faces:
       cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 10)
            
    plt.imshow(img)
    plt.show()