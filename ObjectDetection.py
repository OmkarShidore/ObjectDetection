# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 23:33:09 2019

@author: Omkar Shidore

Setup:
    pip intsall opencv-python
    pip install numpy-python
"""

import cv2
import numpy as np

#Object Created for from frontalface and eye haarcascade
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade =cv2.CascadeClassifier('haarcascade_eye.xml')
hand_cascade=cv2.CascadeClassifier('haarcascade_hand.xml')
palm_cascade=cv2.CascadeClassifier('haarcascade_palm.xml')
img1=np.zeros((128,270,3),np.uint8)

#if prints True, ckechk the path is valid of CascadeClassifier
print(cv2.CascadeClassifier.empty(face_cascade))
print(cv2.CascadeClassifier.empty(eye_cascade))
print(cv2.CascadeClassifier.empty(hand_cascade))
print(cv2.CascadeClassifier.empty(palm_cascade))

capture=cv2.VideoCapture(0)
while True:
    ret,frame=capture.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    hands=hand_cascade.detectMultiScale(gray,1.2,5)
    palms=palm_cascade.detectMultiScale(gray,1.2,5)
    
    for (hx,hy,hw,hh) in hands:
        cv2.rectangle(frame,(hx,hy),(hx+hw,hy+hh),(255,0,0),2)
        cv2.rectangle(gray,(hx,hy),(hx+hw,hy+hh),(0,0,0),2)
    for (px,py,pw,ph) in palms:
        cv2.rectangle(frame,(px,py),(px+pw,py+ph),(255,0,0),2)
        cv2.rectangle(gray,(px,py),(px+pw,py+ph),(0,0,0),2)
    
    for(x,y,w,h) in faces:
        #to draw rectangle, agruments are: image,start points,end_points of rectngle, color of rectangel,line_thickness
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,255,255),2)
        
        #ROI:Rectangle On Image, its creted so the eyes are detected within the rectagle on Face
        roi_gray= gray[y:y+h, x:x+w]
        roi_frame=frame[y:y+h, x:x+w]
        
        #Object eyes
        eyes=eye_cascade.detectMultiScale(roi_gray,1.1,5)
        for (ex,ey,ew,eh) in eyes:
            
            #rectagle frames on eye
            cv2.rectangle(roi_frame,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
            cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,0,0),2)
    
    
    #Printing on Window
    i=5
    j=70
    font_size=1
    font_type=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img1,"Object Detection", (i, j), font_type, font_size,(255,255,255),2)
    
    
    #Showing Image
    
    cv2.imshow('GrayScale Detection',gray)
    cv2.imshow('Colored_Detection',frame)
    cv2.imshow('Project Name',img1)
    
    #waitKey is to close the image window created while running the program, int 27 represent 'Esc' Key
    if cv2.waitKey(1)==27:
        break
capture.release()
cv2.destroyAllWindows()
            
        