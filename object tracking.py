import cv2
import numpy as np
import torch

# YOLOv5 modelini yükleyin (pretrained COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Video dosyasını yükleyin
video = cv2.VideoCapture(r"C:\Users\90552\Downloads\4K Road traffic video for object detection and tracking - free download now!.mp4")

# Sanal çizgi y pozisyonu
line_position = 400
car_count = 0

# Araçların sınıfı
vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

# Araç takip ve sayma fonksiyonu
def detect_and_count_cars(frame):
    global car_count
    results = model(frame)
    detected_objects = results.xyxy[0].numpy()
    
    for obj in detected_objects:
        class_id = int(obj[5])
        class_name = results.names[class_id]

        if class_name in vehicle_classes:
            x1, y1, x2, y2 = map(int, obj[:4])
            confidence = obj[4]

            if confidence > 0.5:
                # Araç için dikdörtgen çizin
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Araç sanal çizgiyi geçtiyse say
                center_y = (y1 + y2) // 2
                if line_position - 10 < center_y < line_position + 10:
                    if is_counted(center_y):
                        car_count += 1

    return frame

# Daha önce sayılıp sayılmadığını kontrol eder
def is_counted(center_y):
    # Her zaman yeni araç olarak say
    return True

# Video işleme döngüsü
while True:
    ret, frame = video.read()
    
    if not ret:
        break
    
    # Araçları tespit edip say
    frame = detect_and_count_cars(frame)
    
    # Sanal çizgiyi çizin
    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (255, 0, 0), 2)
    
    # Araç sayısını ekrana yazdırın
    cv2.putText(frame, f"Total Car Count: {car_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Çıktı penceresi
    cv2.imshow("Araç Takibi", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kaynakları serbest bırakın
video.release()
cv2.destroyAllWindows()
    