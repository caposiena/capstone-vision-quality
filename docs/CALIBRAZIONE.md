# Calibrazione e validazione

Questo documento riassume il percorso seguito per arrivare alla baseline V1.

La calibrazione non è stata fatta scegliendo soglie a caso. Abbiamo osservato il comportamento del modello sui video reali e confrontato i risultati con un conteggio manuale dei difetti.

Il video 2 è stato usato per la calibrazione. Oltre alle detection frame per frame abbiamo introdotto un tracking leggero e un CSV con un solo responso finale per ogni track. Per i difetti di cottura abbiamo aggiunto una conferma basata su HSV e Lab, mentre per i taralli rotti abbiamo usato anche indicatori geometrici come circolarità, solidità, rapporto tra larghezza e altezza ed extent. La deadline è stata regolata osservando il video per congelare il responso prima delle zone più disturbate da riflessi o luce non uniforme.

Sul video 2 il conteggio manuale era: cotto 6, poco 5, blob 7, rotto 14. Il sistema ha restituito: cotto 6, poco 5, blob 6, rotto 15.

A quel punto abbiamo smesso di modificare il sistema sul video 2, per evitare di adattarlo troppo a un singolo caso.

Il video 3 è stato usato come prima validazione indipendente. Il conteggio manuale era: cotto 2, poco 3, blob 1, rotto 8. Il sistema ha restituito: cotto 2, poco 3, blob 1, rotto 9.

Il video 1 presentava uno zoom diverso e condizioni meno uniformi nella parte alta dell'immagine, con anche un'ombra iniziale. Abbiamo quindi adattato solo ROI e deadline, lasciando congelate le soglie di classificazione. Sul difetto rotto il conteggio manuale era 14, mentre il sistema ne ha segnalati 17. Lo scostamento è stato considerato accettabile nel contesto del test e ha evidenziato un limite reale: zoom e illuminazione influenzano la parte geometrica del riconoscimento.

Dopo i tre video la baseline V1 è stata congelata. Addestramento e calibrazione sono considerati conclusi per questa fase del Capstone. Eventuali modifiche future dovranno essere motivate da errori sistematici osservati su nuovi dati, non dal singolo caso.

Per un futuro utilizzo industriale sarà importante lavorare con camera fissa, distanza costante e illuminazione controllata.
