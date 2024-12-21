import os
import cv2
import json
import numpy as np
from ultralytics import YOLO

model = YOLO("best-ocr.pt")

test_patents_folder = "detections"

VALID_CHARACTERS = "BCDFGHJKLPRSTVWXYZ0123456789"

ID_TO_CHAR = {i: char for i, char in enumerate(VALID_CHARACTERS)}

approved_detections = []

def process_patents(folder_path):
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"Error al leer la imagen {img_name}.")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = model.predict(image, conf=0.002, verbose=False)

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

        approved_detections.append({
            "image_name": img_name,
            "recognized_text": recognized_text
        })

        print(f"{img_name}: {recognized_text}")

def save_approved_detections(output_file):
    with open(output_file, "w") as file:
        json.dump(approved_detections, file, indent=4)
    print(f"Detecciones aprobadas guardadas en {output_file}")

process_patents(test_patents_folder)

save_approved_detections("approved_detections.json")
