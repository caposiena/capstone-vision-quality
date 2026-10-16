**Dataset**

Il dataset V2 contiene immagini reali di taralli annotate in formato YOLO. Ogni bounding box ha una sola classe tra buono, cotto, poco, blob e rotto.

**Composizione**

Le immagini complessive sono 96, per un totale di 1127 oggetti annotati.

buono: 330  
cotto: 208  
poco: 169  
blob: 133  
rotto: 287

La suddivisione usata per il training è la seguente:

train: 58 immagini, 736 oggetti  
validation: 15 immagini, 186 oggetti  
test: 23 immagini, 205 oggetti

Il test set proviene da un'acquisizione precedente rispetto a train e validation. È quindi più difficile e permette di osservare meglio cosa succede quando cambiano le condizioni di ripresa.

Quando più frame provengono dallo stesso video, sequenze quasi identiche non vengono distribuite tra train e validation. In questo modo si evita che immagini molto simili finiscano contemporaneamente nei due insiemi.

**Metriche**

Validation interna: precision 0.942, recall 0.972, mAP50 0.980, mAP50-95 0.613.

Test indipendente: precision 0.466, recall 0.823, mAP50 0.546, mAP50-95 0.289.

La differenza tra validation e test è il motivo per cui YOLO viene usato come detector ma il responso finale considera anche tracking, colore e forma.

Nel progetto è stato dato più peso al richiamo dei difetti. Un numero limitato di falsi positivi è accettabile perché i pezzi scartati possono essere controllati dall'operatore.
