# Presentación
Proyecto final de machine learnign en relación al reconocimiento de patentes vehículares mediante OCR usando YOLO11 

# Contenido
- main.py: Posee toda la lógica de aplicación implemnetada en los 2 reconocimientos con FastAPI
- test_patent_local.py: Posee lógica local de reconocimiento de patentes trabajado
- test_ocr_local.py: Posee lógica local de reconocimiento de caractéres trabajado 

# Ejecutar
- main.py: python -m uvicorn main:app --reload
- test_patent_local: python test_patent_local.py
- test_ocr_local: python test_ocr_local.py


# Comentarios
- Este repostiorio unicamente posee la lógica de aplicación.
- **Descargar docker real para probar**: hub.docker.com/r/kiru27/reconocimiento-patentes-ml/tags
