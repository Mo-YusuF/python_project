import cv2

video = cv2.VideoCapture(r"/home/aiktc/Desktop/workshop/imgpro/faceDetection.mp4")
face_cascade = cv2.CascadeClassifier(r"/home/aiktc/Desktop/workshop/imgpro/classifiers/haarcascade_frontalface_default.xml")

print(type(video))

while(True):
    check, frame = video.read()
    #frame_count +=1

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.25, minNeighbors =20)

    for x, y, w, h in faces:
        frame = cv2.rectangle(frame, (x,y), (x+w,y+w), (0,255,0), 3)

    
    cv2.imshow("1st frame of video", frame)
    key = cv2.waitKey(1)
    if(key == ord('q')):
        break

cv2.destroyAllWindows()
video.release()