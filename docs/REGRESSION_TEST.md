# Regression test - video 1, 2, 3

Questo test serve a verificare il comportamento della pipeline dopo le modifiche applicative.

## Regole

- Eseguire i tre video senza modificare soglie o codice tra una prova e l'altra.
- Usare il profilo coerente con il video.
- Assegnare lotti distinti, ad esempio `VIDEO1`, `VIDEO2`, `VIDEO3`.
- Conservare per ogni prova:
  - report lotto;
  - eventi scarto;
  - Audit POCO;
  - conteggi visualizzati nella dashboard.

## Riferimenti già documentati

### Video 2 - calibrazione
Conteggio manuale:
- cotto: 6
- poco: 5
- blob: 7
- rotto: 14

Baseline software storica:
- cotto: 6
- poco: 5
- blob: 6
- rotto: 15
- totale scarti: 32

### Video 3 - validazione indipendente
Conteggio manuale:
- cotto: 2
- poco: 3
- blob: 1
- rotto: 8

Baseline software storica:
- cotto: 2
- poco: 3
- blob: 1
- rotto: 9
- totale scarti: 15

### Video 1 - robustezza
Il riferimento manuale disponibile e' solo:
- rotto: 14

La baseline software storica aveva segnalato:
- rotto: 17

Gli altri conteggi del video 1 non devono essere trattati come ground truth manuale.

## Criterio di audit POCO

Il file `Audit POCO` contiene una riga per ogni track che e' stato visto almeno una volta come `poco`.
Per i casi che finiscono `buono` verificare soprattutto:
- numero osservazioni poco;
- numero osservazioni poco forti;
- confidenza media;
- massimo Lab L;
- responso finale;
- Lab L finale.

Questo permette di distinguere:
- oscillazione del classificatore;
- evidenza colore insufficiente;
- track frammentato/cambiato;
- errore di decisione finale.
