import os
import cv2
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from typing import List
from ultralytics import YOLO

app = FastAPI()

patent_model = YOLO("best-patent.pt")
ocr_model = YOLO("best-ocr.pt")

os.makedirs("detections", exist_ok=True)

CONFIDENCE_THRESHOLD = 0.88
VIDEO_PATH = r"D:\usuarios\diego\escritorio\PROGRA\PYTHON\Tesis\API\prueba2.mp4"
TEST_PATENTS_FOLDER = "detections"
VALID_CHARACTERS = "BCDFGHJKLPRSTVWXYZ0123456789"

ID_TO_CHAR = {i: char for i, char in enumerate(VALID_CHARACTERS)}

detections_list = []

#====================DETECION PATENTES============================#
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el archivo de video: {video_path}")

    saved_detections = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = patent_model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        for detection in results[0].boxes:
            if detection.conf > CONFIDENCE_THRESHOLD:
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

        ret, jpeg = cv2.imencode('.jpg', annotated_frame)
        yield jpeg.tobytes()

    cap.release()

@app.get("/process-patent/")
def process_video_endpoint():
    if not os.path.exists(VIDEO_PATH):
        raise HTTPException(status_code=404, detail=f"El archivo de video {VIDEO_PATH} no existe.")

    video_stream = process_video(VIDEO_PATH)
    return StreamingResponse(
        video_stream,
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

#============================PROCESAMIENTO OCR====================#
@app.get("/process-ocr/")
async def process_patents() -> List[dict]:
    global detections_list

    detections_list = []

    for img_name in os.listdir(TEST_PATENTS_FOLDER):
        img_path = os.path.join(TEST_PATENTS_FOLDER, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Error al leer la imagen {img_name}.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = ocr_model.predict(image, conf=0.002, verbose=False)

        detected_characters = []

        for result in results[0].boxes:
            xyxy = result.xyxy[0].cpu().numpy()
            confidence = result.conf.cpu().numpy()
            class_id = int(result.cls.cpu().numpy())

            if np.any(np.isnan(xyxy)) or np.isnan(confidence):
                continue

            if class_id in ID_TO_CHAR:
                x1 = int(xyxy[0])
                character = ID_TO_CHAR[class_id]
                detected_characters.append((x1, character, confidence))

        detected_characters.sort(key=lambda x: x[0])

        if len(detected_characters) < 6:
            print(f"{img_name}: DESCARTADA (menos de 6 caracteres detectados)")
            continue

        best_6_characters = sorted(detected_characters[:6], key=lambda x: -x[2])
        recognized_text = "".join(char for _, char, _ in best_6_characters)

        detections_list.append({
            "image_name": img_name,
            "recognized_text": recognized_text
        })

        print(f"{img_name}: {recognized_text}")

    return detections_list

@app.get("/saved-detections/")
async def save_approved_detections(output_file: str = "approved_detections.json"):
    detections = await process_patents()

    with open(output_file, "w") as file:
        json.dump(detections, file, indent=4)

    return {"message": f"Detecciones aprobadas guardadas en {output_file}"}
