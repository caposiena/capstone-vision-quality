**Calibrazione e validazione**

La calibrazione è stata eseguita sui video acquisiti sulla linea di prova. Il video 2 è stato usato come riferimento principale, mentre il video 3 è stato tenuto separato per la verifica successiva.

Il responso non dipende soltanto dalla classe restituita da YOLO nell'ultimo frame. Le detection dello stesso pezzo vengono associate in un track e valutate prima della deadline. Per cotto e poco vengono considerati anche i descrittori HSV e Lab. Per rotto entrano nel controllo anche circolarità, solidità, aspect ratio ed extent.

La deadline è stata collocata prima della zona in cui diventano più evidenti riflessi e variazioni di illuminazione. ROI e deadline cambiano in funzione dell'inquadratura; le soglie del classificatore rimangono invece invariate.

**Risultati**

Video 2, conteggio manuale: cotto 6, poco 5, blob 7, rotto 14.  
Video 2, sistema: cotto 6, poco 5, blob 6, rotto 15.

Video 3, conteggio manuale: cotto 2, poco 3, blob 1, rotto 8.  
Video 3, sistema: cotto 2, poco 3, blob 1, rotto 9.

Il video 1 è stato usato come prova di robustezza perché presenta uno zoom diverso e un'illuminazione meno uniforme. Per questo video è disponibile un conteggio manuale verificato solo per la classe rotto: 14 pezzi. Il sistema ne restituisce 17.

Il risultato del video 1 mostra che la componente geometrica è sensibile alla distanza di ripresa e alle condizioni di luce. Su una linea reale sono quindi importanti camera fissa, distanza costante, illuminazione controllata e prodotto disposto su singolo strato.

Dopo il confronto sui tre video le soglie della baseline V1 sono state congelate.
