# Esperimento PC - OpenCV puro / singolo frame

Branch sperimentale: `opencv-single-frame-pc`.

Obiettivo: verificare se il controllo taralli puo funzionare a velocita normale del video senza YOLO e senza tracking temporale.

## Principio

Ogni frame viene trattato come una fotografia indipendente:

1. lettura frame con OpenCV;
2. ROI sul nastro;
3. separazione taralli / fondo blu;
4. morfologia binaria;
5. `findContours`;
6. misure geometriche;
7. misure colore Lab;
8. classificazione deterministica;
9. visualizzazione immediata.

Nessun ID persistente, nessuna deadline decisionale, nessun accumulo temporale.

## Classi sperimentali

- **rotto**: bassa solidity / circularity;
- **blob**: area molto superiore alla mediana degli oggetti del frame;
- **cotto**: Lab L basso;
- **poco**: Lab L alto;
- **buono**: geometria e colore dentro gli intervalli.

Le soglie sono volutamente regolabili live.

## Ambiente PC

Pensato per Windows 10 / Python 3.13.x e l'ambiente del progetto principale.

Dipendenze necessarie per questo esperimento:

```
opencv-python>=4.10
numpy
```

Non servono Ultralytics, Torch, CUDA o TensorRT per questa prova.

## Avvio

Dal repository:

```powershell
python opencv_single_frame.py "C:\percorso\video.mp4"
```

Finestre:

- `visionTar - OpenCV single frame`: risultato;
- `visionTar - Maschera`: segmentazione bianco/nero;
- `visionTar - Settaggi`: soglie e ROI.

Tasti:

- `SPAZIO`: pausa/riprendi;
- `Q` o `ESC`: termina.

## Default ROI

- sinistra 30%;
- destra 12%;
- alto 7%;
- basso 51%.

## Cosa misurare nel test

La prova e positiva solo se:

- il video resta vicino alla velocita originale;
- i taralli sono separati stabilmente nella maschera;
- rotto/blob sono visivamente coerenti;
- cotto/poco rispondono al colore reale;
- le soglie restano ragionevolmente stabili lungo il video.

Questo branch e sperimentale e non modifica il flusso canonico YOLO del `main`.
