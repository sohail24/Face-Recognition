import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read , frame = webcam.read()
    #to apply the alreaday trained haarcascade model we will convert the real time video frames to grayscale
    grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #to identify face , we will use face coordinates
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    #face_coordinates contains 4 tuples x,y for the starting rectange and w and h for the dimension of rectange
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),5)
    #last 2 statements is for color and thickness to place in the screen
    cv2.imshow("Sohail's face detection using haar cascade", frame)
    key = cv2.waitKey(1)
    #argument in waitKey will tell to refresh at x ms
    if (key == 81 or key == 113):
        break

webcam.release()
cv2.destroyAllWindows()