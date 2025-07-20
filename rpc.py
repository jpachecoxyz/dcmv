import cv2
from ultralytics import YOLO
import time
# import RPi.GPIO as GPIO

# Setup del GPIO.
# 20 julio 2025. Aun no cuento con el RasperryPy, por lo cual las lineas siguientes
# Estan comentadas.

# RELAY_PIN = 17  # Usa el GPIO que tengas conectado al relé
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(RELAY_PIN, GPIO.OUT)
# GPIO.output(RELAY_PIN, GPIO.HIGH)  # Energía habilitada por defecto

# Cargar modelo YOLO (nano = más rápido y liviano)
model = YOLO('yolov8n.pt')

# Coordenadas de la celda de trabajo (ajústalas a tu zona real)
# cell_area = (100, 100, 400, 400)  # (x1, y1, x2, y2)

# Celda personalizada
cell_area = (234, 96, 446, 408)

# Función para verificar si el centro de la persona está dentro de la celda
def is_inside_cell(box, cell):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cell[0] <= cx <= cell[2] and cell[1] <= cy <= cell[3]

# Estado anterior para evitar prints innecesarios
previous_state = None  # 'inside' o 'clear'

# Inicializar cámara
# El numero dependera si es la webcam u otro dispositivo, generalmente es 0 o 1.
cap = cv2.VideoCapture(0)  # 0 = cámara USB

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detección silenciosa
    results = model(frame, verbose=False)[0]

    # Revisar si hay alguna persona dentro de la celda
    person_inside = False
    for box in results.boxes:
        if int(box.cls[0]) == 0:  # clase 0 = persona
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if is_inside_cell((x1, y1, x2, y2), cell_area):
                person_inside = True
                break

    # Control del relé y mensajes según el estado
    if person_inside and previous_state != 'inside':
        # GPIO.output(RELAY_PIN, GPIO.LOW)  # Cortar energía
        print("⚠️ Persona detectada: Maquina inhabilitada")
        previous_state = 'inside'

    elif not person_inside and previous_state != 'clear':
        # GPIO.output(RELAY_PIN, GPIO.HIGH)  # Activar energía
        print("✅ Zona libre: Maquina esta en condiciones de ser operada.")
        previous_state = 'clear'


    # Dibujo del rectángulo con nuevo color y grosor
    cv2.rectangle(frame,
                  (cell_area[0], cell_area[1]),
                  (cell_area[2], cell_area[3]),
                  (0, 0, 255),  # Rojo intenso
                  2)

    cv2.imshow("Monitoreo de Celda", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Limpiar todo al salir
cap.release()
cv2.destroyAllWindows()
# GPIO.cleanup()
