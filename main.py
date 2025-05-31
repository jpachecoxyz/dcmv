import cv2
from ultralytics import YOLO

# Carga el modelo YOLOv8 preentrenado
model = YOLO("yolov8n.pt")  # Usa yolov8n por ser liviano

# Define el área de interés (ejemplo: un rectángulo)
area_top_left = (100, 100)
area_bottom_right = (300, 300)

# Captura de la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    person_detected = False

    # Dibujar área de interés
    cv2.rectangle(frame, area_top_left, area_bottom_right, (255, 0, 0), 2)

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # clase 0 = persona
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Dibujar la caja de la persona
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Verifica si el centro de la persona está en el área
            if (area_top_left[0] < cx < area_bottom_right[0] and
                area_top_left[1] < cy < area_bottom_right[1]):
                person_detected = True

    # Mostrar mensaje si se detecta
    if person_detected:
        cv2.putText(frame, "¡Persona en el area!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Detección", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
