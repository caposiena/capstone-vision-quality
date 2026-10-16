# Sistema di visione artificiale per il controllo qualità di prodotti da forno

Capstone EPICODE dedicato al controllo visivo di taralli su nastro trasportatore.

Il prototipo riconosce i singoli pezzi, assegna una classe e registra gli scarti. Le classi utilizzate sono `buono`, `cotto`, `poco`, `blob` e `rotto`.

La pipeline utilizza YOLO per la detection e OpenCV per acquisizione, colore, geometria e visualizzazione. Il risultato viene consolidato con un tracking leggero; il responso viene fissato al passaggio della deadline impostata nella ROI.

## Avvio

Ambiente di sviluppo: Python 3.11.

Installazione dipendenze:

```powershell
python -m pip install -r requirements.txt
```

Avvio dashboard:

```powershell
streamlit run .\dashboard.py
```

Il modello addestrato usato dal progetto è `best.pt` prodotto dal training sul dataset V2.

## File principali

- `visionTar.py`: analisi video, tracking, classificazione finale e generazione dei CSV
- `dashboard.py`: interfaccia Streamlit
- `acquisition_config.py`: profili ROI e deadline
- `video_source.py`: gestione delle sorgenti video
- `database.py`: sessioni, scarti e report SQLite
- `trainTar.py`: training YOLO
- `testTar.py`: valutazione del modello
- `buildDatasetV2.py`: preparazione dataset

Documentazione:
- `docs/DATASET.md`
- `docs/CALIBRAZIONE.md`

## Funzionamento

Flusso principale:

```text
video -> ROI -> YOLO -> tracking -> colore/geometria -> deadline
      -> responso finale -> coordinate scarto -> database/report
```

Per ogni sessione vengono generati:
- CSV delle detection frame per frame;
- CSV con un record per ogni track finalizzato;
- CSV degli scarti con coordinate, bounding box, timestamp e botole associate.

La dashboard registra inoltre data, prodotto e lotto e produce il report riepilogativo `report_lotti.csv`.

## Validazione

La calibrazione è stata eseguita sul video 2. Il video 3 è stato usato come verifica indipendente; il video 1 come prova aggiuntiva con zoom e illuminazione differenti.

Risultati di riferimento:

| Video | Cotto | Poco | Blob | Rotto |
|---|---:|---:|---:|---:|
| 2 - manuale | 6 | 5 | 7 | 14 |
| 2 - sistema | 6 | 5 | 6 | 15 |
| 3 - manuale | 2 | 3 | 1 | 8 |
| 3 - sistema | 2 | 3 | 1 | 9 |

Sul video 1 il conteggio manuale disponibile per `rotto` è 14; il sistema ne rileva 17.

I dettagli della validazione sono in `docs/CALIBRAZIONE.md`.

## Dashboard e simulazione scarto

La dashboard consente di impostare ROI, deadline, larghezza del nastro e numero di botole. La configurazione usata nel prototipo è 40 cm con 20 botole.

Quando un pezzo viene finalizzato come difettoso, la bounding box viene proiettata sulla larghezza utile del nastro e convertita nelle botole interessate. Il file tecnico di scarto conserva anche coordinate in pixel e normalizzate, frame e timestamp.

Il pannello botole rappresenta la selezione calcolata dal software. La sincronizzazione fisica tra camera e attuatore non è compresa nel Capstone e richiederà l'encoder della linea.

## Limiti attuali

La validazione è stata eseguita su video registrati. La lettura da videocamera USB è predisposta per la calibrazione, ma l'elaborazione continua con hardware reale non è stata ancora validata.

Encoder, attuatore e integrazione con il gestionale non fanno parte della versione consegnata.
