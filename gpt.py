import imutils
import cv2
import numpy as np
import time

FRAMES_TO_PERSIST = 10
MIN_SIZE_FOR_MOVEMENT = 200
MOVEMENT_DETECTED_PERSISTENCE = 100

source = 0
cap = cv2.VideoCapture(source)

# Init frame variables
first_frame = None
next_frame = None

# Detector de personas (HOG + SVM preentrenado)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

font = cv2.FONT_HERSHEY_SIMPLEX
delay_counter = 0
movement_persistent_counter = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

while True:
    transient_movement_flag = False
    ret, frame = cap.read()
    text = "Unoccupied"
    
    if not ret:
        print("CAPTURE ERROR")
        continue

    original_frame = frame.copy()
    frame = imutils.resize(frame, width=750)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray

    delay_counter += 1
    if delay_counter > FRAMES_TO_PERSIST:
        delay_counter = 0
        first_frame = next_frame

    next_frame = gray
    frame_delta = cv2.absdiff(first_frame, next_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_people = []

    for c in cnts:
        if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = frame[y:y+h, x:x+w]

            # Detectar personas dentro del ROI del movimiento
            people, _ = hog.detectMultiScale(roi, winStride=(8, 8), padding=(8, 8), scale=1.05)

            for (px, py, pw, ph) in people:
                detected_people.append((x + px, y + py, pw, ph))  # ajustar posición global

    if detected_people:
        transient_movement_flag = True
        for (x, y, w, h) in detected_people:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if transient_movement_flag:
        movement_persistent_flag = True
        movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

    if movement_persistent_counter > 0:
        text = "Person Detected " + str(movement_persistent_counter)
        movement_persistent_counter -= 1
    else:
        text = "No Person Detected"

    cv2.putText(frame, str(text), (10, 35), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("frame", frame)

    # grabación
    if movement_persistent_counter > 0 and not out:
        height, width, _ = frame.shape
        out = cv2.VideoWriter(f'./recordings/{int(time.time())}_video.avi', fourcc, 25.0, (width, height))
        out.write(frame)

    if movement_persistent_counter > 0 and out:
        out.write(frame)

    ch = cv2.waitKey(1)
    if ch & 0xFF == ord('q'):
        if out is not None:
            out.release()
            out = None
        break

cv2.destroyAllWindows()
if out:
    out.release()
cap.release()

