# TFM: Seguimiento multiobjeto de ratones en vista cenital


Este repositorio implementa un pipeline sencillo y reproducible para detección y seguimiento 
multiobjeto en entornos controlados (ratones en visión cenital con cámara fija y fondo estable). 
El sistema combina segmentación binaria con posprocesado morfológico, extracción de regiones (blobs),
asociación temporal basada en filtro de Kalman (modelo de velocidad constante) y asignación Húngara con coste geométrico (1−IoU).
Se incluye un descriptor ligero de apariencia tipo "pixel-pair fingerprint" para trazabilidad visual y refuerzo de identidad.


## Características
- Segmentación por umbral (estática o adaptativa) y limpieza morfológica.
- Etiquetado de componentes y extracción de cajas (ROIs).
- Separación selectiva por watershed en contactos prolongados (si procede).
- Seguimiento con filtro de Kalman y asociación Húngara usando coste 1−IoU.
- Descriptor de apariencia (pixel-pair) para inspección y, opcionalmente, en el coste de asociación.
- Scripts modulares y ejecutable principal `main.py`.


## Requisitos
Instala las dependencias mínimas:

```bash
pip install -r requirements.txt
```
Paquetes clave: `numpy`, `opencv-python` (o `opencv-python-headless`), `scikit-image`, `scipy`, `tqdm`.


## Estructura del proyecto
```
TFM-main/
  main.py                  # Orquesta la lectura de vídeo, detección y seguimiento
  detector_segment.py      # Segmentación, morfología y extracción de blobs/ROIs
  tracker_idtracker.py     # Kalman + Húngaro con coste 1−IoU (estilo idTracker)
  descriptor_pixelpair.py  # Descriptor ligero de apariencia tipo fingerprint
  blob.py                  # Utilidades para blobs/medidas
```


##
Ejecuta el script principal sobre un vídeo de entrada:
```bash
python main.py
```


## Salidas
- Vídeo anotado y/o frames con cajas e IDs (si está habilitado en `main.py`).
- Logs con métricas básicas 
- Carpeta de depuración con instantáneas del pipeline 


## Notas
- Si ejecutas en servidor sin GUI, usa `opencv-python-headless`.
- Para reproducibilidad, fija la tasa de FPS y el tamaño de frame si el vídeo de entrada varía.
- Si integras el descriptor de apariencia en el coste, normaliza adecuadamente la escala de `1−IoU` y del término de apariencia.


## Licencia
Este proyecto se publica con licencia académica para fines docentes/investigación.
