# Dataset

Il dataset usato per la baseline V1 contiene immagini reali di taralli annotate in formato YOLO.

## Classi definitive

Le cinque classi usate dal modello sono:

- `buono`
- `cotto`
- `poco`
- `blob`
- `rotto`

Ogni bounding box riceve una sola etichetta.

## Composizione V2

Il dataset V2 contiene 96 immagini e 1127 oggetti annotati:

- buono: 330
- cotto: 208
- poco: 169
- blob: 133
- rotto: 287

Split utilizzato:

- train: 58 immagini, 736 oggetti
- validation: 15 immagini, 186 oggetti
- test: 23 immagini, 205 oggetti

Il test contiene immagini provenienti da una precedente acquisizione diversa da quella usata per train e validation. Questo lo rende più difficile ma utile per misurare il comportamento fuori dalle condizioni più controllate.

## Criteri di acquisizione

L'acquisizione ideale prevede camera fissa, distanza costante, sfondo coerente, illuminazione stabile, prodotto su singolo strato e orientamento casuale.

Quando i frame provengono dallo stesso video, sequenze quasi identiche non devono essere distribuite tra train e validation, per evitare data leakage.

## Risultati del modello

Sulla validation interna il modello ha ottenuto complessivamente:

- precision: 0.942
- recall: 0.972
- mAP50: 0.980
- mAP50-95: 0.613

Sul test indipendente:

- precision: 0.466
- recall: 0.823
- mAP50: 0.546
- mAP50-95: 0.289

La differenza tra validation e test è stata uno dei motivi per cui il progetto non si è fermato alla sola metrica YOLO. La pipeline video usa anche tracking, conferme colore e supporto geometrico prima di produrre il responso finale.

## Filosofia di errore

Per l'applicazione prevista è preferibile tollerare un numero limitato di falsi positivi, verificabili dall'operatore tra gli scarti, piuttosto che lasciare passare prodotti non conformi.
