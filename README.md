# Sistema di visione artificiale per il controllo qualità di prodotti da forno

## Sinossi

Il progetto nasce da una reale esigenza aziendale legata alla valutazione di un investimento per automatizzare il controllo qualità e la gestione dello scarto di prodotti da forno.

Attualmente il controllo della conformità viene effettuato visivamente da un operatore. L’obiettivo è sviluppare un prototipo software capace di svolgere automaticamente questa funzione su un flusso continuo di prodotto disposto su nastro trasportatore.

Il sistema dovrà acquisire immagini e video tramite telecamera, individuare i singoli prodotti e riconoscere le principali anomalie oggi rilevate manualmente, in particolare difetti di cottura, forma e sovrapposizione.

Per lo sviluppo verranno utilizzati Python, OpenCV per l’acquisizione e l’analisi delle immagini, NumPy per la gestione numerica dei dati, YOLO per il riconoscimento e la localizzazione degli oggetti e Label Studio per la preparazione e l’annotazione del dataset di addestramento.

Per ogni prodotto rilevato il sistema dovrà restituire la classe assegnata, il livello di confidenza e la posizione nell’immagine, così da distinguere i prodotti conformi da quelli da scartare.

Il modello verrà addestrato con una logica prudenziale, privilegiando la riduzione dei falsi negativi, cioè dei prodotti non conformi lasciati passare, anche a costo di accettare un numero limitato di falsi positivi successivamente verificabili dall’operatore.

Il progetto includerà inoltre una dashboard realizzata con Streamlit, collegata a un database leggero, per monitorare gli scarti, associarli ai lotti e alle fasce orarie di produzione, registrare eventuali correzioni dell’operatore e tenere traccia dei test funzionali del sistema di controllo.

La dashboard permetterà inoltre di raccogliere informazioni sui fermi della linea e sulle relative causali, in modo da affiancare al controllo qualità anche un primo livello di monitoraggio della produttività.

Il flusso video verrà utilizzato soltanto per l’elaborazione e non verrà archiviato in modo permanente. Saranno invece conservati i dati sintetici utili alla valutazione del processo produttivo e delle prestazioni del modello.

Il prototipo verrà valutato attraverso metriche di precision, recall, F1-score, qualità della localizzazione e tempi di inferenza, verificandone la compatibilità con le esigenze produttive.

L’integrazione con hardware industriale dedicato, encoder, sistemi di scarto, dispositivi edge e collegamenti al gestionale aziendale resterà fuori dal perimetro del Capstone, pur essendo prevista come possibile evoluzione successiva del progetto.


## Baseline validata V1 - 15 ottobre 2026

Configurazione congelata: ROI sinistra 30%, destra 12%, alto 20%, basso 25%, deadline X 35%. Restano attivi tracking, CSV detection frame-by-frame, CSV finali one-row-per-track, supporto colore per cotto/poco e supporto geometrico per rotto.

Risultati video 2, usato per calibrazione: ground truth manuale cotto 6, poco 5, blob 7, rotto 14; sistema cotto 6, poco 5, blob 6, rotto 15.

Risultati video 3, prima validazione indipendente: ground truth manuale cotto 2, poco 3, blob 1, rotto 8; sistema cotto 2, poco 3, blob 1, rotto 9.

Decisione: baseline V1 in freeze preliminare. Non modificare soglie colore, geometria, tracking o deadline sulla sola base dei video 2 e 3.

Prossimo gate: video 1. Poiche presenta uno zoom/framing differente, ROI e deadline saranno verificati visivamente prima del test. Eventuali regolazioni geometriche specifiche del video 1 non devono modificare le soglie di classificazione senza nuova evidenza.
