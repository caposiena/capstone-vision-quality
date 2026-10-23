Questa cartella ospita i modelli generati localmente per l'inferenza.

Nel progetto principale il modello YOLO atteso e:

`models/best.pt`

Nel branch `opencv-single-frame-pc`, dopo l'esecuzione di:

`python train_opencv_classifier.py`

vengono generati localmente:

- `opencv_extratrees.joblib`
- `opencv_extratrees.json`

Il classificatore ExtraTrees del branch OpenCV lavora solo sui singoli pezzi nelle classi:

- buono
- cotto
- poco
- rotto

La classe blob viene gestita dal gate strutturale sulla maschera runtime e non dal modello ML.

I file di modello non sono versionati nella repository.
