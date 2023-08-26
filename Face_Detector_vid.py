import cv2
from random import randrange
trained_face_data=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  
# classifier is detector                   will detect front faces

webcam=cv2.VideoCapture(0)
# (0) capture video from webcam
# if we give name instead of 0, it will look for file


# looping over all frames until the video ends
while True:
    successful_frame_read, frame=webcam.read()
    # returns two thinngs, 1st; reading from framework was success or not(boolean), 2nd; actaul img


    grayscaled_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # converting img into grayscale


     # detect faces using algorithm we imported
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)


     # looping through faces - to detect every face
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,  (x,y),  (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 10)


    cv2.imshow('face detector', frame)
    # image will pop up


    key=cv2.waitKey(1)
    # it will wait for 1 sec then automatically go for next iteration/next frame

    # Quit if Q key is pressed
    if key==81 or key==113:
        break

# release the video capture object
webcam.release()























