**Sistema di visione artificiale per il controllo qualità di prodotti da forno**

Capstone EPICODE dedicato al controllo visivo di taralli su nastro trasportatore.

Il prototipo riconosce i singoli pezzi, assegna una classe e registra gli scarti. Le classi usate sono buono, cotto, poco, blob e rotto.

La parte di riconoscimento utilizza YOLO. OpenCV viene usato per l'acquisizione dei frame, per i controlli su colore e geometria e per la visualizzazione. Le detection dello stesso prodotto vengono raccolte in un track e il responso viene fissato quando il pezzo raggiunge la deadline impostata nella ROI.

**Avvio**

L'ambiente usato per lo sviluppo è Python 3.11.

Installazione dipendenze: python -m pip install -r requirements.txt

Avvio dashboard: streamlit run .\dashboard.py

Il modello addestrato usato dal progetto è best.pt, ottenuto dal training sul dataset V2.

**File principali**

visionTar.py contiene l'analisi video, il tracking, il responso finale e la generazione dei CSV. dashboard.py contiene l'interfaccia Streamlit. acquisition_config.py raccoglie i profili ROI e deadline. video_source.py gestisce le sorgenti video. database.py gestisce sessioni, scarti e report SQLite.

trainTar.py viene usato per il training, testTar.py per la valutazione del modello e buildDatasetV2.py per la preparazione del dataset.

La documentazione del dataset e della validazione si trova in docs/DATASET.md e docs/CALIBRAZIONE.md.

**Funzionamento**

Il flusso è questo:

video → ROI → YOLO → tracking → colore/geometria → deadline → responso finale → coordinate scarto → database/report

Durante l'uso normale la dashboard conserva il report di lotto e gli eventi di scarto necessari al controllo. Il CSV dettagliato delle detection frame per frame resta disponibile come strumento diagnostico, ma non viene generato nella modalità operativa a basso utilizzo di spazio. Gli eventi di scarto contengono coordinate, bounding box, timestamp e botole associate.

La dashboard registra anche data di produzione, prodotto e lotto e aggiorna il report riepilogativo report_lotti.csv.

**Validazione**

Il video 2 è stato usato per la calibrazione, il video 3 per la verifica indipendente e il video 1 come prova aggiuntiva con zoom e illuminazione differenti.

Video 2, conteggio manuale: cotto 6, poco 5, blob 7, rotto 14.  
Video 2, sistema: cotto 6, poco 5, blob 6, rotto 15.

Video 3, conteggio manuale: cotto 2, poco 3, blob 1, rotto 8.  
Video 3, sistema: cotto 2, poco 3, blob 1, rotto 9.

Sul video 1 il riferimento manuale disponibile per rotto è 14; il sistema ne rileva 17.

I dettagli sono riportati in docs/CALIBRAZIONE.md.

**Dashboard e simulazione dello scarto**

Dalla dashboard si impostano ROI, deadline, larghezza del nastro e numero di botole. La configurazione usata nel prototipo è un nastro da 40 cm con 20 botole.

Quando un pezzo viene classificato come difettoso, la bounding box viene riferita alla larghezza utile del nastro e trasformata nelle botole interessate. Nel file tecnico rimangono anche le coordinate in pixel e normalizzate, il frame e il timestamp.

Il pannello laterale mostra le botole che il software assocerebbe allo scarto. La sincronizzazione fisica tra camera e attuatore non fa parte di questa versione e verrà affidata all'encoder della linea.

**Limiti attuali**

La validazione è stata fatta su video registrati. La videocamera USB è già gestita per la fase di calibrazione, ma l'elaborazione continua con hardware reale non è stata ancora validata.

La velocità di elaborazione dipende dall'hardware utilizzato. I test del Capstone sono stati eseguiti su un PC CPU-only non recente, sul quale l'analisi può risultare più lenta della durata reale del video. La dashboard riduce la frequenza degli aggiornamenti grafici senza saltare frame nell'analisi; per un impiego in linea è prevista una piattaforma di inferenza più adatta al tempo reale.

Encoder, attuatore e collegamento al gestionale restano fuori dal perimetro della versione consegnata.
