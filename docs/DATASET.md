# Dataset

Il dataset V2 contiene immagini reali di taralli annotate in formato YOLO.

Ogni oggetto appartiene a una delle cinque classi: **buono, cotto, poco, blob, rotto**.

## Composizione

Totale: **96 immagini, 1127 oggetti**.

| Classe | Oggetti |
|---|---:|
| buono | 330 |
| cotto | 208 |
| poco | 169 |
| blob | 133 |
| rotto | 287 |

Split utilizzato nel progetto:

| Split | Immagini | Oggetti |
|---|---:|---:|
| train | 58 | 736 |
| validation | 15 | 186 |
| test | 23 | 205 |

Il test set proviene da un'acquisizione differente rispetto a train e validation ed è quindi più severo.

## Metriche del training

Validation interna:

- precision 0.942
- recall 0.972
- mAP50 0.980
- mAP50-95 0.613

Test indipendente:

- precision 0.466
- recall 0.823
- mAP50 0.546
- mAP50-95 0.289

La differenza tra validation e test conferma che il cambio di inquadratura e illuminazione incide in modo importante. Per questo nel sistema finale YOLO non lavora da solo: il responso viene stabilizzato con tracking e controlli OpenCV su colore e forma.

## Benchmark tecnico della soglia operativa

La soglia finale è stata scelta considerando il compito reale del sistema: distinguere prodotti difettosi da prodotti buoni.

Con `conf=0.30`, sul dataset completo di 96 immagini:

- precision difetti: 83.57%
- recall difetti: 95.73%
- F1 difetti: 89.24%
- difetti persi nel benchmark: 34
- falsi scarti nel benchmark: 150

Il confronto con le altre soglie mostra che 0.30 mantiene un richiamo dei difetti molto alto senza portare il numero dei falsi scarti al livello della soglia 0.25.

Questo benchmark comprende anche dati di training e validation. È quindi usato per scegliere la soglia operativa, mentre il comportamento del sistema completo viene verificato separatamente sui video reali descritti in docs/CALIBRAZIONE.md.
