# import cv2 as cv
# import numpy as np
# from matplotlib import  pyplot as plt


# src = cv.imread("/Users/runge.liu/Downloads/INRIAPerson/Train/pos/crop001135.png")
# cv.imshow("input",src)

# hog = cv.HOGDescriptor()
# hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

# #Detect people in image

# (rects,weight) = hog.detectMultiScale(src,winStride=(8,8), padding=(0,0), scale=1.05, useMeanshiftGrouping=False)
# for (x,y,w,h) in rects:
#     cv.rectangle(src,(x,y),(x+w,y+h),(0,255,0),2)

# cv.imshow("hog-detector",src)
# cv.waitKey(0)
# cv.destroyAllWindows()

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 14:47:20 2018
@author: kuangyongjian
test.py
"""
import numpy as np
import cv2 as cv
 
img = cv.imread('/Users/runge.liu/Downloads/INRIAPerson/Train/pos/crop001135.png')

hog = cv.HOGDescriptor()
hog.load('myHogDector1.bin')

#hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
# svm2 = cv.ml.SVM_load('myHogDector1.bin')

# svm2.load("svmtest.mat")
# print(svm2)
# '''开始预测'''
# _, y_pred = svm2.predict(X_test)

cv.imshow('src',img)
cv.waitKey(0)
 
(rects,weight) = hog.detectMultiScale(img,winStride = (8,8),padding = (0,0),scale = 1.05)#,useMeanshiftGrouping=False)
print(rects)
for (x,y,w,h) in rects:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv.imshow("hog-detector",img)
cv.waitKey(0)
cv.destroyAllWindows()


