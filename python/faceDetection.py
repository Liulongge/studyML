import cv2

camera = cv2.VideoCapture(0)
cv2.namedWindow('MyCamera')
face_cascade = cv2.CascadeClassifier(r"/Users/runge.liu/miniforge3/envs/ml/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"/Users/runge.liu/miniforge3/envs/ml/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml")

while True:
    success, frame = camera.read()
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)


    for x, y, w, h in faces:
        img = cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0),2)

        face_area = img[y: y+h, x: x+w]
        eyes = eye_cascade.detectMultiScale(face_area)

        for ex, ey, ew, eh in eyes:
            cv2.rectangle(face_area, (ex,ey), (ex + ew, ey + eh),(0, 255, 0),1)
    cv2.imshow('MyCamera',frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cv2.destroyWindow('MyCamera')
camera.release()

'''
img = cv2.imread("C:\\Users\\runge.liu\\Desktop\\test.jpg")

faces = face_cascade.detectMultiScale(img, 1.3, 5)


for x, y, w, h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0),2)

    face_area = img[y: y+h, x: x+w]
    eyes = eye_cascade.detectMultiScale(face_area)

    for ex, ey, ew, eh in eyes:
        cv2.rectangle(face_area, (ex,ey), (ex + ew, ey + eh),(0, 255, 0),1)

cv2.imshow("im3", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output.jpg", img)
'''
