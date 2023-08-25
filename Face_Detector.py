import cv2
from random import randrange
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  
# classifier is detector                   will detect front faces
# this algorithm detects grayscale images


img=cv2.imread('Robert.webp')
# reading the image into 2 dimensional numbers


grayscaled_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# converting img into grayscale


# detect faces using algorithm we imported
face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)
# multiscale means whatever the scale is smaller or bigger, detect it anyways, detected objects are returned as rectangles


# draw rectangle around the faces
# the two coordinates allow us to draw rectangle over a face

# (x,y,w,h)=face_coordinates[0]   #will automatically assign 4 coordinates to x,y,w,h
# this is for one detecting first face come across


# looping through faces - to detect every face
for (x,y,w,h) in face_coordinates:
    # cv2.rectangle(img,  (x,y),  (x+w, y+h), (0,255,0), 10)
#                 top left        low right  green clr  thickness of rectangle

# assigning different random colors to different faces/ or different color to same face everytime
    cv2.rectangle(img,  (x,y),  (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 10)


print(face_coordinates)
# first two cordinates are upper left corner, last two conrdinates are bottom right corner


cv2.imshow('face detector', img)
# image will pop up

cv2.waitKey()
# wait key pauses the execution of code/ wait until any key is pressed otherwise it will close it instantly

print("Code Completed")