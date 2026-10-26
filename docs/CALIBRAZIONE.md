# Calibrazione e validazione

Il sistema usa YOLO per localizzare e classificare i taralli e OpenCV per gestire video, ROI, tracking, descrittori di colore e geometria, deadline e visualizzazione.

La soglia YOLO operativa è **0.30**. È stata scelta privilegiando il richiamo dei difetti rispetto alla riduzione assoluta dei falsi scarti.

| Confidence | Precision difetti | Recall difetti | F1 difetti |
|---|---:|---:|---:|
| 0.25 | 81.62% | 96.36% | 88.38% |
| 0.30 | 83.57% | 95.73% | 89.24% |
| 0.35 | 84.73% | 95.36% | 89.73% |
| 0.40 | 85.60% | 94.73% | 89.93% |
| 0.45 | 86.62% | 94.23% | 90.26% |

La soglia 0.30 mantiene un richiamo elevato dei difetti e riduce i falsi scarti rispetto a 0.25.

## Responso finale

Il responso non viene preso da una singola detection.

Le osservazioni dello stesso prodotto vengono raccolte in un track. Prima della deadline il sistema combina:

- classe e confidence YOLO;
- descrittori HSV e Lab per i difetti di cottura;
- descrittori geometrici per i prodotti rotti;
- continuità temporale del track.

Quando il prodotto attraversa la deadline, il responso viene fissato.

Per le classi **cotto** e **poco**, YOLO individua il candidato mentre OpenCV conferma il gradiente di colore. Questa scelta riduce i falsi positivi legati a normali variazioni di luminosità.

## Prova video finale

Sul video principale di circa tre minuti il sistema ha analizzato **1.307 prodotti**.

Il riferimento manuale per i difetti di cottura era:

- cotto: 5
- poco: 5

Con la taratura finale il sistema ha restituito:

- cotto: 5
- poco: 5

Le classi geometriche sono state valutate separatamente durante la stessa prova e hanno mostrato un comportamento coerente con l'osservazione del video.

È stato inoltre utilizzato un secondo video di controllo, con **1.129 prodotti analizzati**. Su questo video la classe cotto è rimasta stabile, mentre la classe poco è risultata inizialmente più severa. La soglia colore di poco è stata quindi ribilanciata mantenendo invariati YOLO, tracking, rotto e blob.

Questa seconda prova è servita a evitare una taratura valida soltanto su una singola sequenza.

## Condizioni di utilizzo

Le acquisizioni del Capstone sono state eseguite su video registrati. La configurazione industriale prevista è più controllata:

- camera fissa;
- distanza costante dal prodotto;
- illuminazione boxata;
- schermatura dalla luce ambientale;
- prodotto su singolo strato;
- hardware dedicato all'inferenza;
- sincronizzazione con encoder e attuatori.

Queste condizioni non vengono usate per gonfiare i risultati del prototipo, ma rappresentano il contesto corretto per una successiva validazione industriale.

## Prestazioni

La dashboard offre una modalità completa e una modalità veloce con inferenza su un frame ogni due. La modalità veloce è quella proposta per la demo perché riduce il carico CPU mantenendo timestamp, tracking e logica di responso.

Il prototipo dimostra l'intera catena software, mentre la verifica del vero tempo reale industriale resta legata all'hardware dedicato e all'integrazione fisica sulla linea.
