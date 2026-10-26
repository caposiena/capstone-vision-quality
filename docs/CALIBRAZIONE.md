# Calibrazione e validazione

La configurazione finale usa YOLO per detection e classificazione e OpenCV per video, ROI, tracking, colore, geometria e deadline.

La confidence YOLO usata nel runtime è 0.30.

La soglia è stata scelta privilegiando il richiamo dei difetti rispetto alla riduzione assoluta dei falsi scarti.

Nel benchmark tecnico sul dataset completo:

- conf 0.25: precision 81.62%, recall 96.36%, F1 88.38%
- conf 0.30: precision 83.57%, recall 95.73%, F1 89.24%
- conf 0.35: precision 84.73%, recall 95.36%, F1 89.73%
- conf 0.40: precision 85.60%, recall 94.73%, F1 89.93%
- conf 0.45: precision 86.62%, recall 94.23%, F1 90.26%

La soglia 0.30 è stata mantenuta perché coerente con l'obiettivo del controllo qualità: ridurre il rischio di lasciar passare un difetto.

## Responso finale

Il responso non deriva da una singola detection.

Le osservazioni dello stesso tarallo vengono raccolte in un track e pesate nel tempo.

Prima della decisione il sistema considera:

- classe e confidence YOLO;
- descrittori HSV e Lab per cotto e poco;
- descrittori geometrici per rotto;
- continuità del track.

Per cotto e poco la presenza del difetto deve essere confermata anche dal profilo colore.

Quando il prodotto raggiunge la deadline, il responso viene fissato.

## Prove video

Le prove su video reali sono state utilizzate per verificare il comportamento dell'intera pipeline e per regolare le soglie operative.

Nel video principale, di circa tre minuti, il sistema ha analizzato 1307 prodotti. Il confronto manuale sui difetti di cottura ha portato a 5 cotto e 5 poco, valori riprodotti dalla configurazione calibrata.

Un secondo video è stato utilizzato come ulteriore controllo. Ha confermato la stabilità della classe cotto e ha portato a una regolazione più equilibrata della classe poco.

Queste prove sono prove di calibrazione e verifica del prototipo, non una certificazione industriale.

## Modalità veloce

La dashboard può lavorare su tutti i frame oppure su un frame ogni due.

La modalità veloce conserva frame number e riferimenti temporali, ma utilizza meno osservazioni e può quindi produrre risultati leggermente diversi rispetto alla modalità completa.

## Limiti

Il progetto è stato verificato su video registrati.

L'installazione industriale prevista utilizzerà condizioni più stabili: camera fissa, distanza costante, illuminazione boxata, schermatura dalla luce ambientale e hardware dedicato.

Encoder e attuatori reali non fanno parte del Capstone.
