# Piano Dataset - visionTar

## Obiettivo

Costruire un primo dataset reale e controllabile per addestrare un modello YOLO capace di individuare i singoli taralli e classificare le principali non conformita visive.

Il dataset deve essere rappresentativo delle condizioni future di utilizzo: prodotto disposto su un singolo strato, orientamento casuale, sfondo coerente e illuminazione il piu possibile stabile.

## Classi iniziali

Per la prima fase si parte da cinque classi:

- `ok`
- `poco_cotto`
- `troppo_cotto`
- `forma_anomala`
- `sovrapposto`

Questa tassonomia e intenzionalmente semplice. Prima di aggiungere ulteriori classi bisogna verificare che le cinque iniziali siano distinguibili in modo sufficientemente stabile.

## Regola di annotazione

Ogni tarallo visibile deve ricevere una bounding box e una classe.

La classificazione deve seguire criteri coerenti:

- `ok`: prodotto visivamente conforme;
- `poco_cotto`: colore chiaramente inferiore allo standard accettabile;
- `troppo_cotto`: colore chiaramente superiore allo standard accettabile;
- `forma_anomala`: geometria o struttura del singolo pezzo fuori standard;
- `sovrapposto`: due o piu pezzi sovrapposti o in una configurazione che deve essere considerata non conforme dal sistema.

I casi dubbi non devono essere forzati. Vanno raccolti separatamente per una revisione successiva delle regole.

## Acquisizione immagini

Le immagini devono essere raccolte in condizioni il piu possibile vicine al futuro sistema:

- camera fissa;
- distanza camera-prodotto costante;
- sfondo uniforme;
- illuminazione stabile;
- prodotto disposto su un singolo strato;
- orientamenti casuali;
- piu lotti di produzione;
- piu fasce orarie;
- variabilita reale del prodotto.

Per il primo test non e necessario disporre della linea definitiva. E invece importante evitare immagini casuali ottenute con condizioni di luce e prospettiva completamente diverse tra loro.

## Strategia iniziale

La prima milestone non richiede migliaia di fotografie manuali.

Si puo partire da brevi video reali e ricavare frame campione a intervalli controllati, evitando di usare frame quasi identici in train e validation.

L'obiettivo della prima iterazione e costruire un dataset abbastanza piccolo da essere verificato manualmente, ma abbastanza vario da permettere un primo training YOLO.

## Split dei dati

Le immagini devono essere divise in:

- training;
- validation;
- test.

La divisione non deve essere fatta casualmente frame per frame quando i frame provengono dallo stesso video.

Per ridurre il rischio di data leakage, immagini provenienti dalla stessa sequenza molto ravvicinata o dallo stesso gruppo di acquisizione devono essere mantenute nello stesso split.

Quando possibile, il test finale deve contenere immagini provenienti da un lotto o da una sessione non usata durante il training.

## Struttura prevista

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Le annotazioni verranno esportate in formato YOLO.

## Prima metrica operativa

Nella prima fase non si cerca il modello perfetto.

Il primo obiettivo e verificare che il modello:

1. individui correttamente i singoli taralli;
2. distingua le classi principali;
3. restituisca bounding box e confidence coerenti;
4. mantenga un recall elevato sulle classi di difetto;
5. sia abbastanza rapido da poter essere integrato successivamente nella pipeline video di `visionTar.py`.

## Filosofia di errore

Per l'applicazione finale il sistema dovra essere prudenziale.

E preferibile tollerare un numero limitato di falsi positivi, successivamente verificabili nella cesta degli scarti, piuttosto che lasciare passare falsi negativi.

Questa priorita dovra essere considerata nella valutazione del modello e nella scelta delle soglie di confidence.

## Prossimo gate

Prima di iniziare il training:

- raccogliere un primo campione di immagini o video reali;
- verificare che le cinque classi siano effettivamente distinguibili;
- fissare le regole di annotazione;
- annotare un piccolo dataset pilota;
- eseguire un primo training YOLO;
- analizzare confusioni e classi problematiche.

Solo dopo questo gate si aumentera il dataset.
