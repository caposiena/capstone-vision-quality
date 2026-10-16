# Sistema di visione artificiale per il controllo qualità di prodotti da forno

Progetto Capstone sviluppato per il Master EPICODE.

L'obiettivo è realizzare un prototipo capace di osservare un flusso di taralli su nastro trasportatore, individuare i singoli prodotti e segnalare le principali non conformità visive.

## Stato del progetto

La fase di addestramento e calibrazione della baseline V1 è conclusa.

Classi attuali:

- `buono`
- `cotto`
- `poco`
- `blob`
- `rotto`

Il sistema combina il detector YOLO con un tracking leggero e con controlli aggiuntivi sul colore e sulla geometria. Il responso viene congelato quando il prodotto attraversa una deadline configurabile all'interno della ROI.

La baseline è stata calibrata sul video 2, verificata sul video 3 e successivamente provata anche sul video 1, che presenta uno zoom e condizioni di illuminazione differenti. Dopo questi test le soglie di classificazione sono state congelate.

Il percorso seguito per la calibrazione e i risultati dei test sono descritti in [docs/CALIBRAZIONE.md](docs/CALIBRAZIONE.md).

La composizione del dataset e le metriche del modello sono riportate in [docs/DATASET.md](docs/DATASET.md).

## Struttura del repository

```text
capstone-vision-quality/
├── README.md
├── requirements.txt
├── visionTar.py
├── dashboard.py
├── acquisition_config.py
├── video_source.py
├── trainTar.py
├── testTar.py
├── buildDatasetV2.py
└── docs/
    ├── CALIBRAZIONE.md
    └── DATASET.md
```

### File principali

`visionTar.py` è l'applicazione usata per l'analisi dei video, il tracking e la produzione dei CSV.

`trainTar.py` contiene il flusso di training YOLO.

`testTar.py` viene usato per la valutazione del modello sul test set.

`buildDatasetV2.py` prepara la struttura del dataset V2.

`dashboard.py` contiene l'interfaccia Streamlit per selezionare la sorgente video, configurare ROI e deadline e avviare l'analisi.

`acquisition_config.py` contiene i profili di acquisizione separati dalle soglie del classificatore.

`video_source.py` astrae la sorgente video e consente di usare file, videocamera USB o stream IP senza cambiare la pipeline.

`database.py` gestisce il database SQLite locale con sessioni di produzione e singoli eventi di scarto.

## Pipeline

Il flusso attuale è:

video → ROI → YOLO → tracking → supporto colore/geometria → deadline → responso finale → CSV

Durante l'esecuzione vengono prodotti tre tipi di file:

- un CSV con le detection frame per frame;
- un CSV finale con una sola riga per ogni track finalizzato;
- un CSV scarti che contiene solo i pezzi difettosi e le coordinate necessarie a un futuro sistema di scarto.

I video vengono usati per l'elaborazione e non fanno parte del repository.

## Risultati sintetici

Sul video 2, usato per la calibrazione, il sistema ha restituito:

- cotto: 6, rispetto a 6 reali;
- poco: 5, rispetto a 5 reali;
- blob: 6, rispetto a 7 reali;
- rotto: 15, rispetto a 14 reali.

Sul video 3, usato come prima validazione indipendente:

- cotto: 2, rispetto a 2 reali;
- poco: 3, rispetto a 3 reali;
- blob: 1, rispetto a 1 reale;
- rotto: 9, rispetto a 8 reali.

Il video 1 ha introdotto volutamente una condizione meno favorevole, con zoom diverso e illuminazione meno uniforme. Sul difetto `rotto` il sistema ha contato 17 pezzi contro 14 del conteggio manuale. Il test è stato mantenuto perché mostra in modo utile un limite reale del prototipo e l'importanza, in una futura installazione industriale, di camera e illuminazione controllate.

## Scelte progettuali

Il sistema è volutamente prudenziale: è preferibile accettare un numero limitato di falsi positivi, che possono essere verificati dall'operatore tra gli scarti, piuttosto che lasciare passare prodotti chiaramente non conformi.

L'integrazione con encoder, sistema fisico di scarto, dispositivi edge e gestionale aziendale resta fuori dal perimetro del Capstone.

## Prossimi passi

La baseline V1 è integrata nella parte applicativa. La dashboard gestisce già due profili di acquisizione validati sui video di prova, permette di calibrare ROI e deadline senza modificare il classificatore ed è predisposta per file video, videocamera USB e stream IP.

La dashboard è focalizzata sull'uso con video registrati, coerente con il perimetro della consegna. L'operatore inserisce data di produzione, descrizione prodotto e lotto, seleziona il video di prova e calibra ROI e deadline sul primo fotogramma. Dalla stessa schermata è possibile avviare l'analisi YOLO: `visionTar.py` viene eseguito in modalità headless e registra nella sessione i difetti finalizzati `cotto`, `poco`, `blob` e `rotto`.

Il report finale scaricabile è `report_lotti.csv`, con data, prodotto, lotto, numero di sessioni, totale scarti e dettaglio per tipologia. Il collegamento a videocamera live, encoder e attuatore resta documentato come sviluppo successivo.


## Predisposizione sistema di scarto

Per ogni difetto finalizzato il sistema produce anche un evento tecnico separato dal report qualità. L'evento conserva bounding box in pixel e coordinate normalizzate, timestamp/frame finale e la porzione trasversale del nastro occupata dal prodotto.

La larghezza utile della ROI può essere suddivisa in un numero configurabile di botole. In base all'estensione verticale della bounding box, il software calcola la prima e l'ultima botola coinvolta e l'elenco delle botole candidate da aprire. Se il prodotto attraversa più zone, vengono indicate più botole adiacenti.

La componente longitudinale non viene trasformata in un comando hardware nel Capstone: frame e timestamp costituiscono il riferimento che un futuro encoder userà per determinare il momento di apertura delle botole.

Il file tecnico scaricabile dalla dashboard è denominato `eventi_scarto_<lotto>_<prodotto>.csv`. Il report qualità per lotto resta invece sintetico e separato.

La dashboard prevede due sorgenti di input: video registrato per la validazione del Capstone e videocamera USB per la futura acquisizione diretta. La modalità videocamera permette già di selezionare l'indice del dispositivo e acquisire un fotogramma di calibrazione; l'elaborazione continua live resta da validare quando sarà disponibile l'hardware.


## Simulatore OpenCV delle botole

La dashboard di produzione è stata resa compatta per un pannello da circa 15 pollici. Il flusso video resta centrale, mentre input/calibrazione e contatori sono disposti lateralmente per limitare lo scroll operativo.

Il prototipo usa OpenCV anche per simulare visivamente il banco di scarto. La configurazione iniziale è un nastro da 40 cm diviso in 20 botole nominali da 2 cm. Quando un track viene finalizzato come difettoso, la bounding box viene mappata sulle zone trasversali interessate e il riquadro video evidenzia in rosso le botole candidate all'apertura. Questa visualizzazione è una simulazione software: la selezione delle botole è reale rispetto alle coordinate calcolate, mentre il momento fisico di apertura sarà affidato all'encoder quando l'hardware sarà disponibile.
