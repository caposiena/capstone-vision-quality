# Calibrazione e validazione

Il sistema finale usa YOLO per localizzare e classificare i taralli e OpenCV per gestire video, ROI, tracking, colore, geometria e deadline.

La soglia operativa YOLO è **0.35**. Sul benchmark tecnico dell'intero dataset questa soglia ha migliorato l'equilibrio tra falsi positivi e falsi negativi rispetto a 0.25:

| Confidence | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.25 | 87.18% | 98.94% | 92.68% |
| 0.35 | 91.27% | 97.43% | 94.25% |

Il modello non è stato riaddestrato durante questo confronto: è migliorata la scelta della soglia operativa e, soprattutto, la qualità della verifica.

## Responso finale

Il responso non viene preso dall'ultima detection disponibile.

Le osservazioni dello stesso prodotto vengono associate in un track. Prima della deadline il sistema considera:

- voti e confidence YOLO;
- descrittori HSV e Lab per cotto e poco;
- descrittori geometrici per rotto;
- continuità temporale del track.

Quando il prodotto attraversa la deadline, il responso viene fissato e non cambia più.

## Verifica sui video

Il video 2 è stato usato come riferimento principale; il video 3 come verifica separata.

**Video 2**

- manuale: cotto 6, poco 5, blob 7, rotto 14
- sistema: cotto 6, poco 5, blob 6, rotto 15

**Video 3**

- manuale: cotto 2, poco 3, blob 1, rotto 8
- sistema: cotto 2, poco 3, blob 1, rotto 9

Il video 1 presenta zoom e illuminazione differenti. Per la classe rotto il conteggio manuale disponibile è 14, mentre il sistema ne rileva 17.

Questi risultati mostrano che il prototipo è promettente ma non ancora equivalente a un sistema industriale certificato. Camera fissa, distanza costante, illuminazione controllata e singolo strato restano condizioni importanti.

## Prestazioni

La demo può essere eseguita in modalità completa oppure in modalità veloce, con inferenza su un frame ogni due. La modalità veloce è ora quella proposta di default nella dashboard perché rende l'esecuzione più fluida su CPU, senza modificare timestamp e logica di responso.

La verifica del vero tempo reale industriale richiede hardware dedicato e sincronizzazione con encoder e attuatori.
