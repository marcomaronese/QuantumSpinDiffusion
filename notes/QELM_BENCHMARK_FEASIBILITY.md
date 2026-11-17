# Confronto con i benchmark QELM: sette punti di fattibilità

Registro di fattibilità al 2026-11-17, aggiornato dopo l'esecuzione del punto 1
su autorizzazione dell'utente. Gli altri sei gruppi non sono stati eseguiti.
La numerazione conserva i sette gruppi discussi con l'utente; non è un elenco
esaustivo di tutti gli esperimenti del paper.

Riferimento: Azzam et al., [Generative Models using Multi-layer Quantum
Extreme Learning Machines](https://arxiv.org/abs/2611.20139), versione 1.
Il loro generatore usa feature di un reservoir quantistico fisso e un readout
classico addestrabile. SpinDiffusion apprende invece canali di ricostruzione
di stati e campiona direzioni mediante il readout coerente. I confronti devono
quindi dichiarare rappresentazione, lettura, budget e metriche comuni.

Questa nota registra la fattibilità tecnica. Protocollo e risultati del punto 1
sono nel [rapporto dedicato](EUCLIDEAN_BENCHMARK_REPORT.md). Le priorità restano nel
[piano autorevole](SPINDIFFUSION_PAPER_PLAN.md); le conclusioni supportate
restano nel [claim ledger](../materials/paper/CLAIMS.md).

## 1. Distribuzioni sintetiche in una e due dimensioni

**Stato: implementato ed eseguito per dodici famiglie incondizionate.**
Il nuovo entry point `tasks.synthetic_benchmark` usa un adattatore sferico
fisso, rende esplicito il troncamento delle code e riporta i campioni nelle
coordinate originali. Sono completati 36 confronti appaiati con DDPM e replay
indipendente: gli stati sono ricostruiti con piccolo errore, mentre la qualità
euclidea resta limitata dalla codifica/lettura provata. Non è una replica dei
numeri QELM o una soluzione validata del problema. La variante sintetica
condizionata richiede ancora una pipeline generica per le classi.

Il successivo [studio della mappa e della lettura](EUCLIDEAN_CHART_REPORT.md),
senza addestramenti inversi, conferma miglioramenti dell'errore con una
variante radiale sinh della stereografica. Restano difetti di struttura a
basso spin; non è ancora una nuova esecuzione del benchmark generativo.

Riferimenti: [codifica a singolo spin](../src/spin_quantum_diffusion/quantum/single_spin.py),
[codifica correlata](../src/spin_quantum_diffusion/quantum/correlated.py).

## 2. Generazione condizionata di immagini

**Stato: già disponibile per MNIST 0 e 1; estensioni necessarie per il
benchmark completo.** La pipeline comprende autoencoder, canali inversi
condizionati dalla cifra, POVM prodotto e decoder delle direzioni misurate.
Il caricamento dei dati, il controller e la valutazione sono configurati
per due classi e immagini 28 x 28. Le dieci cifre e Digits 8 x 8 non sono
selezionabili senza modifiche. La disponibilità della pipeline non garantisce
prestazioni comparabili al paper.

Riferimento: [pilota MNIST binario](../tasks/mnist_binary/README.md).

## 3. Variazione della capacità e della profondità

**Stato: disponibili studi analoghi, non una replica degli assi del paper.**
Il codice permette di variare lo spin locale `j`, i passi inversi e gli
strati dei canali; esiste anche uno studio su registri di uno, due e tre spin.
Uno spin nel sottospazio simmetrico e un qubit del loro reservoir hanno
ruoli diversi. Non basta uguagliarne il numero per ottenere risorse uguali;
ogni nuovo confronto deve dichiarare dimensioni, parametri, aggiornamenti,
costi e rappresentazione/readout comuni.

Riferimenti: [configurazione del modello immagini](../tasks/mnist_binary/model.py),
[protocollo della griglia di spin](protocols/SPIN_GRID_V1.md).

## 4. Valutazione tramite MMD e LeNet-FID

**Stato: disponibile una parte per i dati sintetici; LeNet-FID assente.**
Il punto 1 aggiunge W1/sliced W1 e la discrepanza energy/MMD con stimatore V
per campioni 1D/2D. Non aggiunge la loss MMD al training quantistico o il
valutatore LeNet-FID delle immagini. Le altre metriche comprendono
errori di stato, variazione totale delle probabilità di Born, errori di
ricostruzione, accuratezza diagnostica e dispersione dei pixel. Il
classificatore del pilota non sostituisce LeNet-FID. La nuova valutazione
sintetica resta distinta dal cambiamento della loss di addestramento.

Riferimenti: [valutazione MNIST](../tasks/mnist_binary/__main__.py),
[confronto esteso](../tasks/mnist_binary/evaluate_extended.py).

## 5. Confronto con random Fourier features

**Stato: baseline assente.** Sono disponibili comparatori classici
direzionali, inclusi DDPM e i riferimenti del pilota a due spin, ma non il
generatore RFF del paper. Occorre implementare e valutare quella baseline
con input, uscita, selezione e budget dichiarati prima di un confronto.

Riferimenti: [DDPM classico](../src/spin_quantum_diffusion/classical/diffusion.py),
[comparatori a due spin](../tasks/two_spin_correlated.py).

## 6. Hamiltoniane, reservoir fisso e modalità di condizionamento

**Stato: test non direttamente trasferibili.** Il modello corrente non ha
il reservoir Haar fisso con readout affine del paper, né le sue alternative
Hamiltoniane e le tre modalità di inserimento dell'etichetta. Il controller
MNIST usa la classe per produrre i controlli dei canali. Sono possibili
ablazioni specifiche di SpinDiffusion, ma non costituiscono già i test del
paper e richiedono una definizione separata.

Riferimenti: [controller condizionato](../tasks/mnist_binary/model.py),
[generatori dei canali](../src/spin_quantum_diffusion/quantum/correlated.py).

## 7. Effetto del numero di campioni

**Stato: campionamento già disponibile; confronto di robustezza da definire.**
La pipeline MNIST può generare un numero selezionabile di esiti della POVM
e immagini decodificate. Questo permette studi analoghi dell'errore
campionario, con metriche e repliche dichiarate. Non equivale a variare gli
shot usati per stimare le feature del reservoir: sono due fonti di rumore
diverse, e il secondo esperimento non è presente nella pipeline attuale.

Riferimenti: [comando di campionamento](../tasks/mnist_binary/__main__.py),
[POVM coerente](../src/spin_quantum_diffusion/quantum/coherent_readout.py).
