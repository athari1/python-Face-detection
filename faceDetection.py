#!/usr/bin/python
import cv2
import sys

if len(sys.argv) > 1:
    if sys.argv[1] == 'eyes':
        add_eyes = True
    else:
        add_eyes = False
else:
    add_eyes = False

# Define cascade classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_text = '{0} face'.format(len(faces)) if len(
            faces) == 1 else '{0} faces'.format(len(faces))
	eye_gray = gray[y:y + h, x:x + w]
        eye_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(eye_gray)

        # If we want to see the eyes
        if add_eyes:
            eye_gray = gray[y:y + h, x:x + w]
            eye_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(eye_gray)

            for (ex, ey, ew, eh) in eyes:
		cv2.rectangle(eye_color, (ex, ey),
                              (ex + ew, ey + eh), (255, 255, 0), 2)

        eyes_text = '{0} eye'.format(len(eyes)) if len(
                eyes) == 1 else '{0} eyes'.format(len(eyes))
        face_text += ' - ' + eyes_text

    # Display the number of faces detected
    cv2.putText(frame, face_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

    # Display the resulting frame

    cv2.imshow('Face Detection using a webcam ', frame)
    #sys.stdout.write( frame.tostring() )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
