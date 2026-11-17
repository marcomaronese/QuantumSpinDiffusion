# SpinDiffusion per MNIST 0 e 1

L'encoder convoluzionale impara a comprimere le immagini 28 × 28 in una
coppia di direzioni su S² × S². Per ogni cifra, la media dei proiettori
coerenti dei due spin è lo stato bersaglio. Una MLP classica traduce cifra e
passo temporale in angoli per i canali di collisione del repository. La
generazione parte da `I₁₆/16`, applica i canali appresi e campiona una
coppia di direzioni con una POVM prodotto. Il decoder riceve le direzioni
campionate come unico input.

## Esecuzione

Dalla radice del repository, con l'ambiente del progetto:

```bash
.venv/bin/python -m tasks.mnist_binary train \
  --outdir outputs/experiments/mnist_binary_run

.venv/bin/python -m tasks.mnist_binary sample \
  --checkpoint outputs/experiments/mnist_binary_run/checkpoint.pt \
  --outdir outputs/experiments/mnist_binary_samples \
  --digit 0 --count 32 --seed 42
```

Ometti `--digit` per generare entrambe le cifre. `--count` è il numero di
campioni per cifra. Il comando `train` scarica e verifica i due file di training
dal [mirror CVDF](https://github.com/cvdfoundation/mnist), oppure usa i file
già presenti in `outputs/experiments/mnist_binary_data/`. Il comando `sample`
usa il checkpoint in locale. Ogni invocazione richiede una cartella di output
nuova sotto `outputs/experiments/`.

`generated.png` mostra fino a 12 campioni per riga. Per entrambe le cifre,
la prima riga è dedicata allo 0 e la seconda all'1. `generated.npz` contiene
tutti i campioni, le classi richieste e gli indici degli esiti. In
`comparison.png` le righe sono raggruppate per cifra. Ogni gruppo mostra,
dall'alto, originali di validazione, ricostruzioni dal latente, ricostruzioni
dopo misura, campioni dallo stato bersaglio di validazione e campioni dal
modello. `states.npz` salva stati e probabilità per ripetere il confronto.

## Configurazione esplorativa

- L'addestramento riserva 500 esempi per cifra dal training ufficiale per la
  validazione (seed 20261110). `split.npz` registra gli indici originali. Il
  test ufficiale resta chiuso. `metadata.json` registra hash dei dati e dei
  sorgenti, seed e configurazione prima del primo aggiornamento.
- Budget predefinito di 18 epoche per l'autoencoder (Adam, 0.002), 4 epoche
  per calibrare il decoder congelando l'encoder (Adam, 0.001), batch 128 e
  900 aggiornamenti del modello inverso (Adam, 0.025), seed 7. I valori
  esportati corrispondono all'ultimo aggiornamento.
- Due spin con j = 3/2, diffusione isotropa indipendente con D = 1 e
  Δt = 0.75, tre collisioni inverse da `I₁₆/16` con un'ancilla
  fresca condivisa per collisione e due strati di controlli. La loss inversa
  somma l'errore quadratico di Frobenius finale e 0.15 volte la media sui
  passi intermedi.
- Misura coerente con quadratura Gauss–Legendre di ordine 6 in
  cos θ e 12 azimut equispaziati per spin. Il training con gradiente
  usa il kernel coerente continuo sugli stati puri. La calibrazione del
  decoder, la validazione e la generazione usano la stessa POVM finita.
- `metrics.json` registra MSE delle ricostruzioni prima e dopo la misura,
  distanza di traccia fra stati e distanza di variazione totale fra le
  distribuzioni di Born di validazione e del modello. La griglia mostra
  campioni casuali con un seed fissato.

## Training esteso del 2026-11-11

Il confronto esplorativo riusa gli stessi indici di training e validazione
del pilota. Gli abbinamenti usano i seed 7 e 19. Il pilota già salvato è il
controllo breve per il seed 7. Per il seed 19 si esegue lo stesso controllo.
Per entrambi i seed si addestrano `classic` e `rich` con 80 epoche attraverso
la misura continua, 20 di calibrazione finita, batch 128 e 3.000 aggiornamenti
inversi. Si valuta l'ultimo aggiornamento. Il test ufficiale resta chiuso.

```bash
.venv/bin/python -m tasks.mnist_binary train --preset rich \
  --codec-epochs 80 --readout-epochs 20 --reverse-updates 3000 --seed 7 \
  --outdir outputs/experiments/mnist_binary_rich_run
```

`rich` usa j = 5/2 e sei collisioni di tre strati con accoppiamenti separati
fra ciascuno spin e l'ancilla condivisa. Il tempo totale del forward resta
2,25. Le reti sono più larghe e l'autoencoder aggiunge una loss Dice con peso
0,15 per il primo piano. Il modello inverso usa Adam (0,008) e aggiunge alla
loss 0,5 volte la distanza di Hellinger quadratica fra distribuzioni di Born
del bersaglio di training e dello stato generato. Il decoder riceve un unico
esito congiunto della POVM. Il checkpoint include la configurazione usata.

Si riportano per seed e classe l'accuratezza di un classificatore
diagnostico integrata su tutti gli esiti di Born, la variazione totale
verso il bersaglio, l'MSE di ricostruzione e la dispersione RMS dei pixel
all'interno di ogni classe rispetto alle immagini reali di validazione.
Il classificatore è addestrato sulle immagini originali di training
(seed 31, due convoluzioni, quattro epoche, Adam 0,002, batch 128) ed è
comune alle configurazioni. Per scegliere un checkpoint dimostrativo si
usa l'accuratezza media per classe, a parità la variazione totale media
inferiore. Per il confronto visivo si mostrano le prime dieci estrazioni
per classe con seed 42, in ordine di campionamento.
