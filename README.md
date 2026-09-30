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
