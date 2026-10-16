# Dataset

Il dataset V2 contiene immagini reali di taralli annotate in formato YOLO. Ogni bounding box ha una sola classe tra `buono`, `cotto`, `poco`, `blob` e `rotto`.

## Composizione

| Classe | Oggetti |
|---|---:|
| buono | 330 |
| cotto | 208 |
| poco | 169 |
| blob | 133 |
| rotto | 287 |
| **Totale** | **1127** |

Le immagini complessive sono 96.

| Split | Immagini | Oggetti |
|---|---:|---:|
| train | 58 | 736 |
| validation | 15 | 186 |
| test | 23 | 205 |

Il test set proviene da un'acquisizione precedente rispetto a train e validation. Per questo motivo è più difficile e rappresenta meglio il cambio di condizioni di ripresa.

Quando più frame provengono dallo stesso video, sequenze quasi identiche non vengono distribuite tra train e validation, per evitare che immagini molto simili compaiano in entrambi gli insiemi.

## Metriche

Validation interna:
- precision: 0.942
- recall: 0.972
- mAP50: 0.980
- mAP50-95: 0.613

Test indipendente:
- precision: 0.466
- recall: 0.823
- mAP50: 0.546
- mAP50-95: 0.289

La differenza tra validation e test ha portato a mantenere YOLO come detector, aggiungendo nel flusso video tracking e controlli su colore e forma prima del responso finale.

Nel progetto viene privilegiato il richiamo dei difetti: un numero limitato di falsi positivi è accettabile perché i pezzi scartati possono essere verificati dall'operatore.
