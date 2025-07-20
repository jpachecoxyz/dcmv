import cv2
from ultralytics import YOLO

# Cargar el modelo YOLOv8
model = YOLO("yolov8n.pt")  # o 'yolov5s.pt' si usas torch.hub

# Coordenadas de la celda de trabajo (x1, y1, x2, y2)
cell_area = (100, 100, 400, 400)

def is_inside_cell(box, cell):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return cell[0] <= cx <= cell[2] and cell[1] <= cy <= cell[3]

cap = cv2.VideoCapture(0)  # Usa 0 para cámara o un archivo .mp4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    # Dibujar celda
    cv2.rectangle(frame, (cell_area[0], cell_area[1]), (cell_area[2], cell_area[3]), (255, 255, 0), 2)

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls != 0:  # Solo personas
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_inside = is_inside_cell((x1, y1, x2, y2), cell_area)

        color = (0, 255, 0) if person_inside else (0, 0, 255)
        label = "Person IN" if person_inside else "Person OUT"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("YOLOv8 - Celda de Trabajo", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

