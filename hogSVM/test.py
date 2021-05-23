import cv2 as cv
import numpy as np
from matplotlib import  pyplot as plt


src = cv.imread("/Users/runge.liu/Downloads/INRIAPerson/Train/pos/crop001135.png")
cv.imshow("input",src)

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

#Detect people in image

(rects,weight) = hog.detectMultiScale(src,
                                      winStride=(8,8),
                                      padding=(0,0),
                                      scale=1.05,
                                      useMeanshiftGrouping=False)
for (x,y,w,h) in rects:
    cv.rectangle(src,(x,y),(x+w,y+h),(0,255,0),2)

cv.imshow("hog-detector",src)
cv.waitKey(0)
cv.destroyAllWindows()