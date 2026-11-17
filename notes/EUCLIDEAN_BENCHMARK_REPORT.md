# Benchmark sintetico euclideo v1 — punto 1 del confronto QELM

Completato e verificato il 2026-11-17. Risultato esplorativo, esterno alle evidenze congelate del manoscritto.

Il modello apprende gli stati codificati con errore contenuto, ma questa codifica e lettura a spin finito alterano fortemente le distribuzioni euclidee. Il risultato non dimostra una buona generazione dei dati originali, né un limite universale di SpinDiffusion.

## Protocollo e provenienza

- [Protocollo congelato](protocols/EUCLIDEAN_BENCHMARK_V1.md): dodici famiglie incondizionate, tre realizzazioni ciascuna, seed 1031, 2081 e 4093.
- 36 confronti appaiati: 36 modelli quantistici e 36 DDPM classici. Ogni coppia usa 2048 dati di training e altri 2048 per validazione, test e riferimento indipendente reale. Tutti i fit sono terminati prima della valutazione di test; nessuna selezione o riottimizzazione sui risultati.
- Singolo spin j=5/2, dimensione 6, sei collisioni inverse con tre strati e un'ancilla fresca; 162 parametri, 512 aggiornamenti, loss finale di Frobenius, prior I6/6. Non e' un registro generico di sei qubit come nel paper.
- DDPM direzionale cartesiano esistente: 5635 parametri e 512 aggiornamenti. Gli aggiornamenti coincidono, i costi e le capacita' no. La lettura comune passa anche i campioni DDPM attraverso codifica coerente e POVM; la lettura diretta e' riportata separatamente.
- Mappa fissa con media e scala apprese solo dal training; proiezione radiale entro tre unita' standardizzate e sollevamento sull'emisfero. Decoder affine delle coordinate della direzione misurata, senza rete appresa. La frazione troncata e' dichiarata.
- POVM coerente positiva e completa 24 x 48; 2048 campioni per condizione. W1 in 1D e sliced W1 su 32 direzioni fisse in 2D, in unita' standardizzate sul training. Energy V-statistic secondaria sui primi 1024 campioni: stimatore con diagonali e bias finito, senza test d'ipotesi.
- Le famiglie seguono [le figure 4-5 del paper](https://arxiv.org/html/2611.20139v1); parametri assenti dalle didascalie sono scelte locali esplicite. I numeri non sono una replica confrontabile direttamente con quelli QELM.

Run: `outputs/experiments/euclidean_benchmark_v1_20261117`.

Fingerprint dei sorgenti numerici: `22111a8261302ba039f2a795208b1292545f1dbebadaa090984887c51a146ac9`.
Manifest del run: `7e336c245f20871bbd6f352a8dba3fd3aa169fcf07c80828bf771adc380d82f2`.
Il run conserva sorgenti, protocollo, dati, mappe, checkpoint, traiettorie, probabilita', esiti e metriche. La successiva sistemazione grafica di titoli/assi usa soltanto i campioni salvati in una directory distinta; lo snapshot numerico e il suo manifest rimangono intatti.

## Discrepanza rispetto ai dati originali

Media +/- deviazione standard campionaria su tre realizzazioni; valori inferiori sono migliori. Il riferimento codificato usa lo stato del test soltanto come diagnostica della rappresentazione, mai per addestrare. Non e' un limite inferiore matematico per W1 e le differenze tra queste colonne non sono una scomposizione additiva dell'errore.

| Famiglia | Reale vs reale | Bersaglio codificato | SpinDiffusion | DDPM + POVM comune | DDPM diretto |
|---|---:|---:|---:|---:|---:|
| bimodal | 0.031 +/- 0.024 | 0.548 +/- 0.007 | 0.549 +/- 0.004 | 0.511 +/- 0.012 | 0.363 +/- 0.005 |
| peak_exponential | 0.029 +/- 0.016 | 0.821 +/- 0.009 | 0.827 +/- 0.007 | 0.857 +/- 0.010 | 0.381 +/- 0.011 |
| student_t | 0.060 +/- 0.011 | 0.678 +/- 0.033 | 0.655 +/- 0.050 | 0.638 +/- 0.034 | 0.138 +/- 0.011 |
| gaussian | 0.039 +/- 0.004 | 0.463 +/- 0.018 | 0.476 +/- 0.040 | 0.435 +/- 0.033 | 0.172 +/- 0.006 |
| log_normal | 0.042 +/- 0.014 | 0.792 +/- 0.021 | 0.801 +/- 0.022 | 0.810 +/- 0.006 | 0.243 +/- 0.020 |
| skewed_mixture | 0.025 +/- 0.009 | 0.585 +/- 0.025 | 0.558 +/- 0.014 | 0.550 +/- 0.067 | 0.318 +/- 0.055 |
| correlated_gaussian | 0.044 +/- 0.021 | 0.536 +/- 0.008 | 0.538 +/- 0.005 | 0.522 +/- 0.019 | 0.150 +/- 0.009 |
| banana | 0.042 +/- 0.014 | 0.634 +/- 0.002 | 0.632 +/- 0.004 | 0.639 +/- 0.017 | 0.235 +/- 0.013 |
| anticorrelated_gaussian | 0.041 +/- 0.012 | 0.473 +/- 0.015 | 0.485 +/- 0.013 | 0.468 +/- 0.012 | 0.127 +/- 0.013 |
| funnel | 0.046 +/- 0.003 | 0.553 +/- 0.008 | 0.554 +/- 0.017 | 0.534 +/- 0.023 | 0.166 +/- 0.010 |
| two_moons | 0.031 +/- 0.007 | 0.433 +/- 0.007 | 0.413 +/- 0.020 | 0.396 +/- 0.007 | 0.196 +/- 0.025 |
| ring | 0.026 +/- 0.004 | 0.395 +/- 0.014 | 0.383 +/- 0.008 | 0.375 +/- 0.011 | 0.258 +/- 0.014 |

L'errore di SpinDiffusion resta vicino al riferimento codificato, molto distante dal riferimento reale-vs-reale. Nel bimodale e nell'anello la frazione di test troncata e' zero, quindi il solo taglio delle code non spiega la perdita osservata. I picchi stretti, le curve e il vuoto centrale dell'anello sono gia' degradati dalla codifica/POVM e dal decoder fissati.

Il DDPM diretto ha errore medio inferiore a SpinDiffusion in tutte le dodici famiglie; attraversando la stessa codifica/POVM subisce una forte perdita. Il primo e' un confronto con una pipeline diversa, il secondo un controllo di lettura comune. Nessuno dei due stabilisce efficienza o vantaggio quantistico.

## Ricostruzione dello stato e troncamento

| Famiglia | Distanza di traccia modello-test | TV Born modello-test | Test troncato |
|---|---:|---:|---:|
| bimodal | 0.0121 +/- 0.0094 | 0.0069 +/- 0.0054 | 0.00% |
| peak_exponential | 0.0099 +/- 0.0017 | 0.0050 +/- 0.0014 | 2.36% |
| student_t | 0.0140 +/- 0.0098 | 0.0076 +/- 0.0053 | 1.40% |
| gaussian | 0.0128 +/- 0.0012 | 0.0070 +/- 0.0011 | 0.23% |
| log_normal | 0.0087 +/- 0.0019 | 0.0043 +/- 0.0010 | 1.56% |
| skewed_mixture | 0.0074 +/- 0.0026 | 0.0037 +/- 0.0022 | 0.49% |
| correlated_gaussian | 0.0191 +/- 0.0024 | 0.0097 +/- 0.0043 | 3.35% |
| banana | 0.0183 +/- 0.0098 | 0.0100 +/- 0.0064 | 4.46% |
| anticorrelated_gaussian | 0.0214 +/- 0.0055 | 0.0124 +/- 0.0048 | 2.36% |
| funnel | 0.0169 +/- 0.0015 | 0.0096 +/- 0.0014 | 3.08% |
| two_moons | 0.0178 +/- 0.0018 | 0.0098 +/- 0.0013 | 0.00% |
| ring | 0.0248 +/- 0.0062 | 0.0151 +/- 0.0040 | 0.00% |

La distanza di traccia media su tutti i fit e' 0.01529; la TV media di Born e' 0.00842. Questi valori riguardano gli stati codificati e la POVM dichiarata, non la fedelta' della distribuzione euclidea originale.

## Verifica, griglia e costi

- Replay NumPy/SciPy indipendente: 36 coppie e 316 hash verificati; errore massimo di traiettoria 3.286e-14.
- Minimo autovalore tra gli stati controllati: 1.949e-09; nessuna proiezione PSD o riparazione degli stati.
- Residuo massimo di completezza di Kraus: 2.084e-14; errore massimo del decadimento multipolare: 6.661e-16.
- La verifica indipendente ripete anche codifica, diffusione, completezza della POVM, campionamento Born, decoding, W1, energy e aggregazioni.
- Raffinando 24 x 48 a 48 x 96, la TV degli istogrammi a bordi fissi cambia fino a 0.11334. Le piccole differenze di TV istogrammata non vanno sovrainterpretate.
- Per questo motivo e' stato aggiunto un controllo post-hoc separato, senza fit o nuovi campioni: il trasporto esatto pesato tra le due griglie, nella metrica primaria W1/sliced W1, e' al massimo 0.01642. Questa differenza e' piccola rispetto agli errori euclidei osservati; non e' una prova di convergenza al continuo.
- Su un thread CPU, training medio per fit: 6.38 s quantistico e 0.37 s classico. Tempo del benchmark prima del rendering: circa 250.2 s. Sono costi di simulazione locale, non stime hardware.
- La suite del repository comprende 102 test e passa. I controlli grafici sono stati eseguiti sui PNG salvati.

## Artefatti e ripresa

- [Distribuzioni 1D](../outputs/experiments/euclidean_benchmark_v1_figures_20261117/distributions_1d.png).
- [Distribuzioni 2D](../outputs/experiments/euclidean_benchmark_v1_figures_20261117/distributions_2d.png).
- [Confronto degli errori](../outputs/experiments/euclidean_benchmark_v1_figures_20261117/discrepancy_comparison.png).
- [Audit](../outputs/experiments/euclidean_benchmark_v1_20261117/audit.json) e [risultati aggregati](../outputs/experiments/euclidean_benchmark_v1_20261117/summary.json).
- [Diagnostica di trasporto tra griglie](../outputs/experiments/euclidean_benchmark_v1_grid_transport_20261117/grid_transport.json).
- Questi output sono locali e ignorati da Git. Codice, protocollo e questo rapporto sono versionabili; nessun risultato e' stato promosso alle evidenze del paper.

Il punto 1 e' stato implementato ed eseguito per i casi incondizionati. La qualita' euclidea della configurazione provata rimane insufficiente; il risultato diagnostico identifica un collo di bottiglia nella rappresentazione/lettura scelta, senza stabilire quale modifica lo risolva. Non sono state provate altre mappe, spin o architetture, ne' la necessita' della diffusione. I test appena usati sono consumati: eventuali scelte successive richiedono nuovi split e un protocollo distinto. Il prossimo passo scientifico del progetto resta E5 nel piano autorevole.

## Valori per realizzazione

| Famiglia | Seed | W1 oracle | W1 modello | W1 DDPM comune | Traccia modello-test |
|---|---:|---:|---:|---:|---:|
| bimodal | 1031 | 0.55510 | 0.54512 | 0.50856 | 0.02297 |
| bimodal | 2081 | 0.54040 | 0.54866 | 0.52406 | 0.00633 |
| bimodal | 4093 | 0.54756 | 0.55257 | 0.49960 | 0.00714 |
| peak_exponential | 1031 | 0.81414 | 0.82913 | 0.85074 | 0.01112 |
| peak_exponential | 2081 | 0.83121 | 0.81892 | 0.85240 | 0.00794 |
| peak_exponential | 4093 | 0.81674 | 0.83325 | 0.86829 | 0.01059 |
| student_t | 1031 | 0.64274 | 0.59828 | 0.59876 | 0.02482 |
| student_t | 2081 | 0.68445 | 0.69284 | 0.65470 | 0.00562 |
| student_t | 4093 | 0.70695 | 0.67415 | 0.66029 | 0.01168 |
| gaussian | 1031 | 0.47963 | 0.43168 | 0.39638 | 0.01143 |
| gaussian | 2081 | 0.46400 | 0.50943 | 0.45767 | 0.01345 |
| gaussian | 4093 | 0.44458 | 0.48687 | 0.44987 | 0.01347 |
| log_normal | 1031 | 0.78963 | 0.77585 | 0.81736 | 0.00831 |
| log_normal | 2081 | 0.77252 | 0.81641 | 0.80701 | 0.01078 |
| log_normal | 4093 | 0.81386 | 0.81066 | 0.80704 | 0.00710 |
| skewed_mixture | 1031 | 0.56770 | 0.56375 | 0.51930 | 0.00788 |
| skewed_mixture | 2081 | 0.61455 | 0.54161 | 0.62663 | 0.00468 |
| skewed_mixture | 4093 | 0.57389 | 0.56817 | 0.50459 | 0.00975 |
| correlated_gaussian | 1031 | 0.54286 | 0.53240 | 0.50143 | 0.01871 |
| correlated_gaussian | 2081 | 0.53689 | 0.54191 | 0.52806 | 0.01687 |
| correlated_gaussian | 4093 | 0.52751 | 0.53958 | 0.53787 | 0.02160 |
| banana | 1031 | 0.63450 | 0.62699 | 0.62100 | 0.02959 |
| banana | 2081 | 0.63532 | 0.63241 | 0.65414 | 0.01191 |
| banana | 4093 | 0.63096 | 0.63567 | 0.64112 | 0.01341 |
| anticorrelated_gaussian | 1031 | 0.45491 | 0.47034 | 0.45830 | 0.02368 |
| anticorrelated_gaussian | 2081 | 0.48195 | 0.49196 | 0.48151 | 0.02543 |
| anticorrelated_gaussian | 4093 | 0.48084 | 0.49420 | 0.46528 | 0.01522 |
| funnel | 1031 | 0.56191 | 0.56676 | 0.54209 | 0.01758 |
| funnel | 2081 | 0.54655 | 0.53468 | 0.50895 | 0.01520 |
| funnel | 4093 | 0.54908 | 0.55920 | 0.55210 | 0.01806 |
| two_moons | 1031 | 0.43528 | 0.42963 | 0.39648 | 0.01920 |
| two_moons | 2081 | 0.43843 | 0.39108 | 0.40251 | 0.01849 |
| two_moons | 4093 | 0.42593 | 0.41907 | 0.38932 | 0.01580 |
| ring | 1031 | 0.38202 | 0.39129 | 0.38708 | 0.02952 |
| ring | 2081 | 0.40932 | 0.38345 | 0.37248 | 0.02717 |
| ring | 4093 | 0.39240 | 0.37475 | 0.36535 | 0.01782 |
