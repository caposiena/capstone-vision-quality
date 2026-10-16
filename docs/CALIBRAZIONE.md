# Calibrazione e validazione

La baseline V1 è stata definita mediante prove su video reali e confronto con conteggi manuali dei difetti.

Il video 2 è stato utilizzato come riferimento di calibrazione. La pipeline combina detection frame per frame, tracking leggero e un responso finale univoco per ogni track. Per i difetti di cottura viene utilizzato un supporto basato su descrittori HSV e Lab; per la classe `rotto` vengono considerati anche indicatori geometrici quali circolarità, solidità, rapporto larghezza/altezza ed extent. La deadline è posizionata prima delle zone maggiormente influenzate da riflessi o variazioni di illuminazione.

## Video 2 - calibrazione

Conteggio manuale:
- cotto: 6
- poco: 5
- blob: 7
- rotto: 14

Risultato della baseline:
- cotto: 6
- poco: 5
- blob: 6
- rotto: 15

## Video 3 - validazione indipendente

Conteggio manuale:
- cotto: 2
- poco: 3
- blob: 1
- rotto: 8

Risultato della baseline:
- cotto: 2
- poco: 3
- blob: 1
- rotto: 9

## Video 1 - prova di robustezza

Il video 1 presenta uno zoom differente e condizioni di illuminazione meno uniformi, inclusa un'ombra nella fase iniziale. Per questa prova sono stati adattati ROI e deadline senza modificare le soglie di classificazione.

Riferimento manuale disponibile:
- rotto: 14

Risultato della baseline:
- rotto: 17

Il test evidenzia la sensibilità della componente geometrica a distanza di ripresa, zoom e illuminazione.

## Condizioni di utilizzo

Per una futura installazione industriale sono raccomandati:
- camera fissa;
- distanza di ripresa costante;
- illuminazione controllata;
- prodotto disposto su singolo strato;
- calibrazione coerente della ROI rispetto alla larghezza utile del nastro.

Le soglie di classificazione della baseline V1 restano congelate; eventuali modifiche devono essere motivate da evidenze ripetibili su nuovi dati.
