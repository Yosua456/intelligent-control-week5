from ultralytics import YOLO
import cv2

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi pose
    results = model(frame)

    # Tampilkan hasil dengan anotasi
    for result in results:
        annotated_frame = result.plot()  # Tambahkan anotasi pada frame

    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)

    # Tekan 'q' untuk keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup jendela OpenCV
cap.release()
cv2.destroyAllWindows()
