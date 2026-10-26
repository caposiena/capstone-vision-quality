# Dataset

Il dataset V2 contiene immagini reali di taralli annotate in formato YOLO.

Le classi sono buono, cotto, poco, blob e rotto.

## Composizione

Totale: 96 immagini e 1127 oggetti.

- buono: 330
- cotto: 208
- poco: 169
- blob: 133
- rotto: 287

Split:

- train: 58 immagini, 736 oggetti
- validation: 15 immagini, 186 oggetti
- test: 23 immagini, 205 oggetti

Il test set proviene da un'acquisizione diversa da train e validation.

## Metriche

Validation interna:

- precision: 0.942
- recall: 0.972
- mAP50: 0.980
- mAP50-95: 0.613

Test set:

- precision: 0.466
- recall: 0.823
- mAP50: 0.546
- mAP50-95: 0.289

La differenza tra validation e test mostra la sensibilità del modello al cambio di condizioni di acquisizione.

Per questo il sistema finale non usa soltanto la classificazione YOLO, ma consolida il responso attraverso tracking e controlli OpenCV.

## Benchmark della soglia

Con confidence 0.30, sul dataset completo:

- precision difetti: 83.57%
- recall difetti: 95.73%
- F1 difetti: 89.24%

Il benchmark comprende anche dati di train e validation. Serve quindi alla scelta della soglia operativa e non va considerato un test indipendente.

Le prove video della pipeline completa sono descritte in docs/CALIBRAZIONE.md.
