# visionTar

Sistema di visione artificiale per il controllo qualità dei taralli su nastro trasportatore.

Il progetto nasce come Capstone EPICODE e simula un controllo in linea: il software individua i prodotti, li segue mentre attraversano l'area di ispezione, assegna un responso finale e registra gli scarti.

Le classi sono:

- buono
- cotto
- poco
- blob
- rotto

## Architettura

La pipeline è volutamente ibrida.

**YOLO** individua i singoli taralli e propone la classe con una confidence.

**OpenCV** gestisce il video, la ROI, il tracking, la deadline, i descrittori di colore e forma, la visualizzazione e la simulazione delle botole di scarto.

Le osservazioni dello stesso tarallo vengono raccolte in un track. Il responso non dipende quindi da un singolo frame: viene consolidato prima della deadline e poi congelato.

Flusso essenziale:

```text
video -> ROI -> YOLO -> tracking -> colore/forma -> deadline
      -> responso finale -> coordinate di scarto -> report di lotto
```

## Dashboard

La dashboard Streamlit permette di:

- caricare un video o predisporre una videocamera USB;
- impostare prodotto, lotto, ROI e deadline;
- scegliere la frequenza di analisi;
- vedere il nastro con bounding box, tracking e responso;
- visualizzare i contatori degli scarti per classe;
- simulare l'associazione tra difetto e botole;
- salvare sessioni ed esportare report CSV.

Per la demo la modalità predefinita è **Veloce - 1 frame su 2**. Riduce il carico di inferenza mantenendo i riferimenti temporali del video. La modalità completa resta disponibile.

## Avvio

Ambiente di sviluppo: Python 3.11.

Dopo il clone:

1. creare e attivare un ambiente virtuale;
2. installare le dipendenze con `python -m pip install -r requirements.txt`;
3. copiare il modello addestrato in `models/best.pt`;
4. avviare la dashboard con:

```powershell
streamlit run .\dashboard.py
```

Il modello non è incluso nella repository.

## File principali

- `visionTar.py`: inferenza video, tracking e responso finale;
- `dashboard.py`: interfaccia Streamlit;
- `acquisition_config.py`: profili ROI e deadline;
- `database.py`: sessioni, eventi di scarto e report;
- `trainTar.py`: training YOLO;
- `testTar.py`: valutazione del modello;
- `buildDatasetV2.py`: preparazione del dataset.

La documentazione tecnica essenziale è in `docs/DATASET.md` e `docs/CALIBRAZIONE.md`.

## Risultati

Il dataset V2 contiene **96 immagini e 1127 oggetti annotati**.

Sul benchmark tecnico eseguito sull'intero dataset con soglia di confidence 0.35, YOLO ha ottenuto:

- precision: **91.27%**
- recall: **97.43%**
- F1 detection: **94.25%**
- accuratezza di classe sui match: **91.71%**

Questo benchmark include anche immagini utilizzate durante training e validation, quindi serve a verificare la coerenza del modello e della soglia operativa, non va interpretato come misura indipendente di generalizzazione.

La verifica indipendente resta quella sul test set e sui video separati descritti nei documenti di validazione.

## Limiti

Il prototipo è stato validato su video registrati. La resa dipende molto da camera fissa, illuminazione controllata e prodotto disposto su singolo strato.

La sincronizzazione fisica tra camera, encoder e attuatori non fa parte della versione consegnata. Nel prototipo viene simulata l'associazione tra posizione del difetto e botole di scarto.

Il tempo reale industriale richiede hardware di inferenza adeguato; i test del Capstone sono stati eseguiti anche su PC CPU-only.
