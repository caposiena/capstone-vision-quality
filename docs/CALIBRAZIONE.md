# Calibrazione e validazione

Il sistema finale usa YOLO per localizzare e classificare i taralli e OpenCV per gestire video, ROI, tracking, colore, geometria e deadline.

La soglia operativa YOLO è **0.30**.

La scelta è stata fatta guardando il problema come controllo qualità binario: difettoso contro buono. L'obiettivo principale è non lasciare passare prodotti difettosi; un numero limitato di falsi scarti è considerato meno grave di un difetto non intercettato.

| Confidence | Precision difetti | Recall difetti | F1 difetti |
|---|---:|---:|---:|
| 0.25 | 81.62% | 96.36% | 88.38% |
| 0.30 | 83.57% | 95.73% | 89.24% |
| 0.35 | 84.73% | 95.36% | 89.73% |
| 0.40 | 85.60% | 94.73% | 89.93% |
| 0.45 | 86.62% | 94.23% | 90.26% |

La soglia 0.30 mantiene un richiamo dei difetti molto alto e riduce i falsi scarti rispetto a 0.25. Le soglie superiori privilegiano ulteriormente la precisione, ma iniziano a perdere più difetti.

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
