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

`dashboard.py` contiene la prima dashboard Streamlit per selezionare la sorgente video e calibrare visivamente ROI e deadline.

`acquisition_config.py` contiene i profili di acquisizione separati dalle soglie del classificatore.

`video_source.py` astrae la sorgente video e consente di usare file, videocamera USB o stream IP senza cambiare la pipeline.

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

Con la baseline V1 congelata è iniziata la parte applicativa. La dashboard gestisce già due profili di acquisizione validati sui video di prova, permette di calibrare ROI e deadline senza modificare il classificatore ed è predisposta per file video, videocamera USB e stream IP.

Il prossimo gate è collegare l'inferenza alla dashboard e salvare in un database leggero i dati di sessione, lotto, conteggi, scarti e correzioni dell'operatore.
