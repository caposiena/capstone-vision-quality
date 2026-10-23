Questa cartella ospita i modelli generati localmente per l'inferenza.

Nel progetto principale il modello YOLO atteso è:

`models/best.pt`

Nel branch `opencv-single-frame-pc`, dopo l'esecuzione di:

`python train_opencv_classifier.py`

vengono generati localmente:

- `opencv_extratrees.joblib`
- `opencv_blob_binary.joblib`
- `opencv_extratrees.json`

I file di modello non sono versionati nella repository.
