# Come abbiamo calibrato il sistema

La calibrazione non e stata fatta scegliendo soglie a caso, ma osservando il comportamento del modello sui video reali e confrontandolo con un conteggio manuale dei difetti.

Siamo partiti dal video 2, usato come video di calibrazione. Per ogni classe abbiamo confrontato il numero di difetti riconosciuti dal sistema con quelli contati manualmente: cotto, poco cotto, blob e rotto.

All'inizio il modello riconosceva bene alcuni difetti, ma tendeva a perdere diversi casi di cotto e rotto e a classificare troppi prodotti come poco cotti. Per capire il motivo non ci siamo limitati al risultato finale: abbiamo salvato le detection frame per frame, introdotto un tracking leggero per seguire lo stesso tarallo nel tempo e aggiunto un secondo CSV con un solo responso finale per ogni track.

Per i difetti di cottura abbiamo poi analizzato anche il colore. Sono stati usati descrittori HSV e Lab, in particolare la luminosita Lab L, per distinguere meglio i taralli troppo chiari da quelli troppo scuri. Il colore non sostituisce YOLO, ma viene usato come conferma del responso.

Per i taralli rotti abbiamo invece aggiunto alcuni indicatori geometrici semplici, come circolarita, solidita, rapporto tra larghezza e altezza ed extent. Anche in questo caso la geometria non lavora da sola, ma rafforza il responso quando esiste gia una evidenza del modello.

Un passaggio importante e stato anche il posizionamento della deadline, cioe la linea oltre la quale viene congelato il responso finale. La sua posizione e stata regolata osservando il video reale per evitare sia la perdita dei track sia l'influenza di una zona successiva con riflessi di luce.

Dopo ogni modifica abbiamo ripetuto il test sullo stesso video e confrontato il risultato con il conteggio manuale. Sul video 2 siamo arrivati a:

- cotto: 6 reali, 6 riconosciuti
- poco: 5 reali, 5 riconosciuti
- blob: 7 reali, 6 riconosciuti
- rotto: 14 reali, 15 riconosciuti

A quel punto abbiamo smesso di modificare il sistema sul video 2 per evitare di adattarlo troppo a quel singolo caso.

La configurazione e stata quindi provata sul video 3, che non era stato usato per la calibrazione. Il confronto con il conteggio manuale ha dato:

- cotto: 2 reali, 2 riconosciuti
- poco: 3 reali, 3 riconosciuti
- blob: 1 reale, 1 riconosciuto
- rotto: 8 reali, 9 riconosciuti

Questo secondo test e stato importante per verificare che le regole introdotte non funzionassero solo sul video usato durante la calibrazione.

La configurazione attuale e quindi stata congelata come prima baseline validata. Eventuali prove successive, come il video 1 che presenta uno zoom diverso, serviranno soprattutto a verificare la robustezza della ROI e della deadline senza cambiare nuovamente le soglie di classificazione se non emergono evidenze reali.
