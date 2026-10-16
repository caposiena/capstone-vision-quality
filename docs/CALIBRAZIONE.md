# Calibrazione e validazione

La calibrazione è stata eseguita sui video acquisiti sulla linea di prova. Il video 2 è stato usato come riferimento principale; il video 3 è stato tenuto separato per la verifica successiva.

La decisione finale non usa soltanto la classe restituita da YOLO nell'ultimo frame. Le detection dello stesso pezzo vengono associate in un track e valutate prima della deadline. Per `cotto` e `poco` vengono considerati anche i descrittori HSV e Lab. Per `rotto` entrano nel controllo anche circolarità, solidità, aspect ratio ed extent.

La deadline è stata collocata in una zona dell'immagine precedente ai riflessi più evidenti. ROI e deadline dipendono dall'inquadratura; le soglie del classificatore rimangono invece invariate tra i video validati.

## Risultati

| Prova | Cotto | Poco | Blob | Rotto |
|---|---:|---:|---:|---:|
| Video 2 - manuale | 6 | 5 | 7 | 14 |
| Video 2 - sistema | 6 | 5 | 6 | 15 |
| Video 3 - manuale | 2 | 3 | 1 | 8 |
| Video 3 - sistema | 2 | 3 | 1 | 9 |

Il video 1 è stato utilizzato come prova di robustezza perché presenta zoom diverso e illuminazione meno uniforme. In questo caso è disponibile un conteggio manuale verificato solo per la classe `rotto`: 14 pezzi. Il sistema ne restituisce 17.

Il comportamento del video 1 conferma che la parte geometrica è sensibile a distanza di ripresa e illuminazione. Per un impiego su linea reale servono quindi camera fissa, distanza costante, illuminazione controllata e prodotto disposto su singolo strato.

Dopo il confronto sui tre video le soglie della baseline V1 sono state congelate.
