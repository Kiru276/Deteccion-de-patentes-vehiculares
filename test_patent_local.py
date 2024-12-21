import cv2
import os
from ultralytics import YOLO

model = YOLO("best-patent.pt")

video_path = r'D:\usuarios\diego\escritorio\PROGRA\PYTHON\Tesis\PATENT\prueba2.mp4'
cap = cv2.VideoCapture(video_path)

confidence_threshold = 0.88

os.makedirs("detections", exist_ok=True)

saved_detections = {}

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        results = model.track(frame, persist=True)

        annotated_frame = results[0].plot()

        for detection in results[0].boxes:
            if detection.conf > confidence_threshold:
                detection_id = detection.id
                detection_confidence = detection.conf
                
                if detection_id not in saved_detections or detection_confidence > saved_detections[detection_id]['confidence']:
                    
                    saved_detections[detection_id] = {
                        'confidence': detection_confidence,
                        'image': frame
                    }
                    
                    x1, y1, x2, y2 = map(int, detection.xyxy[0])
                    detected_region = frame[y1:y2, x1:x2]
                    
                    cv2.imwrite(f"detections/id_{detection_id}.jpg", detected_region)
        
        cv2.imshow("YOLO11 Tracking", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
