# visionTar

visionTar è un prototipo di controllo qualità visivo per taralli su nastro trasportatore, sviluppato come Capstone EPICODE.

Il sistema riconosce cinque classi: buono, cotto, poco, blob e rotto.

## Come funziona

YOLO individua i taralli nel frame e propone una classe con relativa confidence.

OpenCV gestisce il video, la ROI, il tracking, la deadline e alcuni controlli aggiuntivi su colore e geometria.

Le osservazioni dello stesso prodotto vengono raccolte durante il movimento sul nastro. Il responso finale viene fissato quando il track raggiunge la deadline.

Per cotto e poco, la classificazione YOLO viene confermata anche attraverso descrittori di colore HSV e Lab. Per rotto vengono utilizzati anche indicatori geometrici.

La soglia YOLO usata nel runtime è 0.30.

## Dashboard

La dashboard Streamlit permette di:

- caricare un video;
- inserire prodotto, lotto e data;
- regolare ROI e deadline;
- scegliere la modalità di elaborazione;
- vedere detection, tracking e responso;
- visualizzare i conteggi per classe;
- simulare la botola interessata dallo scarto;
- salvare ed esportare i risultati di lotto.

La modalità veloce analizza un frame ogni due. Mantiene i riferimenti temporali del video, ma riduce il numero di osservazioni disponibili rispetto alla modalità completa.

## Dataset

Il dataset V2 contiene 96 immagini reali e 1127 oggetti annotati.

Le classi sono:

- buono: 330
- cotto: 208
- poco: 169
- blob: 133
- rotto: 287

Lo split è composto da 58 immagini di train, 15 di validation e 23 di test.

I dettagli sono in docs/DATASET.md.

## Risultati

Sul benchmark tecnico usato per scegliere la soglia operativa, con confidence 0.30, la valutazione binaria difettoso/buono ha restituito:

- precision difetti: 83.57%
- recall difetti: 95.73%
- F1 difetti: 89.24%

Questo benchmark comprende anche immagini di train e validation e non rappresenta quindi una misura indipendente di generalizzazione.

Le prove video sono state usate per verificare il comportamento dell'intera pipeline e per calibrare soprattutto i difetti di cottura.

## Stato del progetto

Il Capstone dimostra l'intera catena software: acquisizione video, detection, tracking, decisione, localizzazione dello scarto e registrazione dei dati.

La parte fisica di espulsione non è implementata: le botole sono simulate.

La futura applicazione industriale prevede camera fissa, illuminazione controllata, encoder, attuatori e hardware dedicato all'inferenza.

## File principali

visionTar.py: motore di inferenza, tracking e responso.

dashboard.py: interfaccia Streamlit.

acquisition_config.py: profili di acquisizione.

database.py: sessioni e report.

trainTar.py: training YOLO.

testTar.py: valutazione del modello.

buildDatasetV2.py: preparazione del dataset.
