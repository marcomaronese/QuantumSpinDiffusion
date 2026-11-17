# SpinDiffusion — difficoltà e soluzioni da considerare

Creato il 2026-11-15.

Questo documento raccoglie i dodici punti discussi e offre uno spazio per
aggiungere progressivamente le soluzioni proposte, le verifiche necessarie e
gli esiti. La numerazione resta stabile per poter richiamare ciascun problema.

Le priorità, le dipendenze e le decisioni sugli esperimenti sono definite nel
[piano autorevole](SPINDIFFUSION_PAPER_PLAN.md). Questo registro raccoglie la
discussione; una proposta annotata qui non equivale a una soluzione verificata
o a un esperimento già programmato. Le conclusioni utilizzabili nel paper
restano quelle del [registro delle evidenze](../materials/paper/CLAIMS.md).

I punti 1–6 descrivono difficoltà osservate o confronti ancora mancanti. Gli
studi recenti a più spin e il pilota MNIST sono risultati di lavoro non ancora
integrati nelle evidenze congelate del paper. I punti 7–12 distinguono vincoli
matematici da rischi la cui incidenza sul modello deve ancora essere verificata.

Per ogni proposta futura, indicare che cosa cambia, perché dovrebbe aiutare,
come verificarla e quali costi o effetti collaterali comporta. Aggiungere gli
esiti con data e riferimenti, conservando anche le proposte scartate e la
motivazione della decisione.

## Difficoltà già riscontrate e confronti mancanti

### 1. Aumentare lo spin migliora la rappresentazione, ma rende più difficile apprenderla

Nello studio a tre spin, aumentando lo spin locale da `j=1/2` a `j=5/2`
diminuisce l'errore introdotto dalla codifica, ma cresce quello del modello
rispetto al bersaglio codificato. La variazione totale media tra dati originali
e lettura dello stato codificato scende da 0,667 a 0,501; quella tra lettura
del bersaglio codificato e del modello sale da 0,011 a 0,295. L'errore
complessivo tra dati originali e modello rimane tra 0,657 e 0,669.

Il risultato riguarda la famiglia di circuiti studiata e il budget fisso di
160 aggiornamenti. Non dimostra che aumentare lo spin sia sempre dannoso,
né confronta configurazioni a parità di costo computazionale.

**Riferimenti:** [protocollo della griglia](protocols/SPIN_GRID_V1.md) e
[risultati nel piano, sessione del 2026-11-11](SPINDIFFUSION_PAPER_PLAN.md#session-log-and-current-status).

- **Soluzioni proposte da considerare:** da aggiungere.
- **Verifiche e criteri di accettazione:** da definire per ciascuna proposta.
- **Esiti e decisioni:** da compilare dopo la valutazione.

### 2. La scelta della funzione di perdita influenza fortemente il risultato

Nel caso a singolo spin, a 100 epoche la supervisione del solo risultato
finale recupera entrambi i modi in 5/5 realizzazioni, contro 1/5 con
supervisione dell'intero percorso. Un fallimento può quindi dipendere
dall'addestramento, senza dimostrare un limite dell'architettura.

La loss di percorso storica somma sei termini. Questi risultati non isolano
una causa specifica, come un conflitto tra gradienti, e non stabiliscono che
la supervisione finale sia sempre preferibile su altri problemi.

**Riferimenti:** [studio diagnostico](DIAGNOSTIC_REPORT.md) e
[claim C4–C5](../materials/paper/CLAIMS.md).

- **Soluzioni proposte da considerare:** da aggiungere.
- **Verifiche e criteri di accettazione:** da definire per ciascuna proposta.
- **Esiti e decisioni:** da compilare dopo la valutazione.

### 3. Migliorare le correlazioni può peggiorare le distribuzioni marginali

Le architetture con ancilla condivisa riducono alcuni errori di correlazione,
ma peggiorano la ricostruzione dei singoli spin e dello stato globale rispetto
ai canali locali indipendenti. Nel confronto di validazione eseguito, nessuna
delle due configurazioni condivise supera i criteri prestabiliti; la relativa
conferma resta chiusa.

Il risultato documenta un compromesso sfavorevole nelle condizioni studiate.
Non dimostra un'impossibilità generale di apprendere insieme marginali e
correlazioni, né identifica da solo il meccanismo responsabile.

**Riferimento:** [rapporto sul controllo locale indipendente](TWO_SPIN_LOCAL_CONTROL_REPORT.md).

- **Soluzioni proposte da considerare:** da aggiungere.
- **Verifiche e criteri di accettazione:** da definire per ciascuna proposta.
- **Esiti e decisioni:** da compilare dopo la valutazione.

### 4. I modelli classici restano più accurati nei confronti disponibili

Nella conferma a due spin, la distanza di traccia media è circa 0,462 per il
modello quantistico selezionato, 0,328 per il modello fattorizzato e 0,027 per
il riferimento classico basato sul kernel di calore. Anche gli altri modelli
classici riportati in quel confronto hanno un errore medio inferiore a quello
del modello quantistico selezionato.

I budget differiscono: questi risultati documentano il divario di qualità
nelle condizioni eseguite, senza stabilire un confronto di efficienza o una
superiorità universale dei metodi classici.

**Riferimenti:** [conferma della supervisione a due spin](TWO_SPIN_SUPERVISION_REPORT.md#confirmation-on-previously-unopened-observations)
e [claim C8 per il precedente pilota](../materials/paper/CLAIMS.md).

- **Soluzioni proposte da considerare:** da aggiungere.
- **Verifiche e criteri di accettazione:** da definire per ciascuna proposta.
- **Esiti e decisioni:** da compilare dopo la valutazione.

### 5. Su MNIST, riconoscibilità e varietà non coincidono

Il modello più ricco produce cifre più riconoscibili secondo il classificatore
diagnostico, ma la dispersione dei pixel all'interno di ogni classe rimane
inferiore a quella dei dati reali di validazione. Riprodurre l'identità della
cifra risulta più facile che riprodurne tutta la diversità.

Il risultato resta esplorativo: riguarda le cifre 0 e 1, due seed abbinati e
lo stesso split di training e validazione. Il test ufficiale MNIST resta
chiuso. La misura osservata non basta a stabilire una diagnosi generale di
collasso dei modi.

**Riferimenti:** [pilota MNIST e configurazioni](../tasks/mnist_binary/README.md)
e [risultati nel piano, sessione del 2026-11-11](SPINDIFFUSION_PAPER_PLAN.md#session-log-and-current-status).

- **Soluzioni proposte da considerare:** da aggiungere.
- **Verifiche e criteri di accettazione:** da definire per ciascuna proposta.
- **Esiti e decisioni:** da compilare dopo la valutazione.

### 6. Il contributo specifico della diffusione resta da isolare

Mancano confronti controllati sufficienti con preparazione diretta dello
stato, un solo canale appreso e catene più corte. I risultati mostrano che si
possono apprendere canali di ricostruzione, ma non dimostrano che tutti i passi
della diffusione siano necessari.

La supervisione del solo risultato finale non impone il recupero degli stati
intermedi del processo diretto. Il suo successo non dimostra quindi, da solo,
un beneficio specifico della diffusione o un recupero dei dettagli secondo
un ordine prestabilito.

**Riferimenti:** [fase E4 del piano](SPINDIFFUSION_PAPER_PLAN.md#e4--test-whether-diffusion-and-the-proposed-structure-add-value),
[discussione del paper](../materials/paper/sections/discussion.tex) e
[affermazioni escluse dal registro](../materials/paper/CLAIMS.md#claims-deliberately-excluded).

- **Soluzioni proposte da considerare:** da aggiungere.
- **Verifiche e criteri di accettazione:** da definire per ciascuna proposta.
- **Esiti e decisioni:** da compilare dopo la valutazione.

## Vincoli e difficoltà individuabili sul piano teorico

### 7. Perdita irreversibile di informazione nella codifica

A spin finito sono accessibili soltanto le armoniche con `ell <= 2j`.
Distribuzioni originali differenti possono produrre lo stesso stato
quantistico: nessun addestramento può distinguerle usando soltanto quello
stato. Anche una ricostruzione perfetta restituisce, attraverso la lettura
coerente adottata, una distribuzione smussata.

Si tratta di un limite matematico della rappresentazione, distinto da un
errore di ottimizzazione. Anche le armoniche mantenute vengono attenuate;
il solo valore del taglio armonico non determina la risoluzione utile.

**Riferimenti:** [teoria della codifica e della lettura](../materials/paper/sections/theory.tex)
e [claim C1–C2 e limiti dichiarati](../materials/paper/CLAIMS.md).

- **Soluzioni proposte da considerare:** da aggiungere.
- **Verifiche e criteri di accettazione:** da definire per ciascuna proposta.
- **Esiti e decisioni:** da compilare dopo la valutazione.

### 8. Instabilità del recupero dei dettagli attenuati

La diffusione sopprime le componenti come `exp[-D ell(ell+1)t]`.
L'inversione matematica dei coefficienti attenuati amplifica gli errori,
in particolare per le componenti più fini. Un canale fisico non può inoltre
invertire universalmente la perdita di distinguibilità introdotta dalla
diffusione.

Il modello inverso deve apprendere informazioni specifiche sulla distribuzione
bersaglio. Preparare uno stato bersaglio a partire dal prior massimamente
misto non equivale a realizzare un'inversione universale del processo diretto.

**Riferimenti:** [spettro della diffusione e canali inversi](../materials/paper/sections/theory.tex)
e [Watrous, contrattività dei canali quantistici](https://johnwatrous.com/wp-content/uploads/TQI.pdf).

#### Proposta 8.1 — Addestramento progressivo per gruppi di armoniche

**Origine:** proposta dell'utente del 2026-11-15.
**Stato:** da considerare; non implementata né verificata.

Allenare prima le componenti più grossolane della distribuzione e introdurre
progressivamente quelle più fini. È un curriculum sulla risoluzione della
loss, applicato inizialmente alla stessa architettura e allo stesso spin.
L'ipotesi è che imparare una struttura grossolana prima dei dettagli renda
l'ottimizzazione più favorevole; non è un beneficio già dimostrato per
SpinDiffusion.

La successiva precisazione dell'utente collega la suddivisione in fasi anche
a una diffusione più debole per ogni gruppo di rango crescente. Questa
proposta descrive la parte di curriculum della loss; la scelta coordinata
della forza diffusiva per gruppo è registrata nella proposta 8.2.

Un esempio per `j=5/2`, dove il rango massimo accessibile è 5:

1. Dare priorità ai ranghi `ell=1,2`.
2. Aggiungere i ranghi `ell=3,4`, mantenendo nella loss anche `ell=1,2`.
3. Aggiungere `ell=5`, mantenendo tutti i ranghi precedenti.
4. Eseguire un eventuale affinamento congiunto con l'obiettivo finale completo,
   contando anche questi aggiornamenti nel budget totale.

I gruppi sono illustrativi: composizione, pesi e durata delle fasi devono
essere fissati nel protocollo oppure selezionati soltanto su validazione.
L'aggiunta di un gruppo può avvenire gradualmente attraverso i suoi pesi.

**Come esprimere la loss.** Con
`c_ell,m(rho) = Tr[T_ell,m^dagger rho]`, una forma candidata alla fase `s` è:

```text
L_s = alpha * L_globale(rho_gen, rho_train)
      + beta * sum_{ell=1}^{2j} w_ell(s)
               * sum_{m=-ell}^{ell} |c_ell,m(rho_gen) - c_ell,m(rho_train)|^2
```

Qui `rho_gen` è sempre l'uscita dell'intera catena generata da `I/d`,
`rho_train` è il bersaglio codificato dai dati di training e i pesi delle
bande non ancora introdotte sono zero. Il rango zero è fissato dalla traccia.
I coefficienti complessi, inclusi fase e orientamento, sono confrontati
direttamente; la sola potenza per rango non basta. I pesi vanno normalizzati
con una convenzione dichiarata, per esempio bilanciando il numero `2ell+1`
di componenti e la somma dei pesi attivi. Questa convenzione evita un aumento
banale della scala della loss dovuto al numero di termini, senza garantire
gradienti uguali tra le fasi. `alpha`, `beta` e il criterio globale devono
essere dichiarati; un termine globale con peso contenuto può accompagnare
la priorità assegnata alle bande attive.

**Accorgimenti e limiti.**

- Conservare una penalità sui ranghi già introdotti: ottimizzare soltanto il
  gruppo nuovo potrebbe distruggere quanto appreso prima. Nella prima
  variante considerata si mantengono allenabili i parametri del circuito;
  congelare blocchi di canali non garantisce la conservazione delle armoniche,
  perché i canali successivi possono modificarle.
- Cambiare i pesi della loss sullo stato finale non richiede di troncare
  artificialmente la matrice bersaglio. Un taglio netto dei multipoli non
  garantisce uno stato positivo; mantenere il bersaglio fisico completo e
  la costruzione CPTP della catena.
- Usare i coefficienti del bersaglio di training, senza ricostruirli mediante
  moltiplicazione per `exp[+D ell(ell+1)t]`. La proposta apprende la
  distribuzione bersaglio; non stabilizza per definizione un inverso universale.
- Il curriculum potrebbe aiutare l'ottimizzazione, ma non recupera armoniche
  escluse dalla codifica, non rende informativi dettagli sotto il rumore
  statistico e non supera i limiti di capacità o di rimozione dell'entropia.
  Potrebbe anche condurre a una soluzione grossolana difficile da affinare:
  l'ordine di apprendimento è una scelta da verificare.
- Le fasi del curriculum sono fasi di addestramento. Non coincidono
  automaticamente con i passi temporali inversi o con blocchi separati del
  circuito. Il fallimento del precedente addestramento temporale su ingressi
  esatti non è una verifica di questa proposta; allo stesso modo, l'eventuale
  successo del curriculum finale non dimostrerebbe un recupero ordinato dei
  ranghi lungo la traiettoria generativa.

**Verifiche e criteri di accettazione da formalizzare.** Confrontare il
curriculum con un controllo che usa tutte le bande dall'inizio e lo stesso
obiettivo finale, mantenendo architettura, spin, dati, inizializzazioni e
numero totale di aggiornamenti abbinati. Contare tutte le fasi e riportare
anche il costo misurato: lo stesso numero di aggiornamenti non implica lo
stesso tempo di calcolo. Misurare l'errore complesso per rango durante tutte
le fasi, la distanza di traccia o fedeltà globale e l'errore della lettura
comune; su più spin includere marginali e correlazioni congiunte. Il beneficio
richiede un miglioramento dei dettagli fini senza un peggioramento oltre le
soglie prestabilite dei ranghi bassi o delle metriche globali, seguito da
conferma indipendente dopo la selezione. Il protocollo dovrà dichiarare
metrica primaria, soglie e identità dei dati prima delle esecuzioni.

**Riferimenti della proposta:** [Bengio et al., Curriculum Learning](https://ronan.collobert.com/pub/2009_curriculum_icml.pdf)
per il principio generale, non come dimostrazione del beneficio su questo
modello; [loss multipolare già disponibile](../src/spin_quantum_diffusion/quantum/losses.py);
[diagnosi della supervisione temporale](TWO_SPIN_SUPERVISION_REPORT.md#validation-selection).

**Esiti e decisioni:** nessun esperimento eseguito e nessun beneficio
confermato. Proposta registrata per una futura valutazione secondo il piano
autorevole; non modifica l'attuale prossima azione E5.

#### Proposta 8.2 — Diffusione più debole per gruppi di rango crescente

**Origine:** precisazioni dell'utente del 2026-11-15: allenare i ranghi `ell`
a gruppi e usare una diffusione progressivamente più debole per i gruppi fini.
**Stato:** proposta chiarita; non implementata né verificata sperimentalmente.

Associare a ogni gruppo di armoniche un compito di ricostruzione con forza
diffusiva adeguata alla sua scala. Questo può limitare uniformemente lo
spegnimento delle componenti nel singolo compito. Non richiede necessariamente
di modificare lo spettro del generatore: si possono mantenere i canali di
calore isotropi standard e variare intensità o durata tra i compiti.

**Scelta quantitativa candidata.** Se il gruppo `b` arriva al rango `L_b`,
indicare con `a_b = D_b * t_b` la forza integrata della diffusione applicata
nel suo compito. Dalla legge del modello:

```text
c_ell,m(dopo) = exp[-a_b * ell(ell+1)] * c_ell,m(prima)
```

Scelto un valore `kappa > 0`, imporre:

```text
a_b = kappa / [L_b * (L_b+1)]
```

La componente di rango massimo del gruppo viene allora moltiplicata sempre
per `exp[-kappa]`, indipendentemente da `L_b`; le altre componenti del gruppo
sono attenuate meno. Per esempio, scegliendo `kappa = log(2)`:

| Gruppo | Forza integrata candidata | Frazione conservata al rango massimo |
|---|---|---|
| `ell=1,2` | `log(2)/6` | `1/2` |
| `ell=3,4` | `log(2)/20` | `1/2` |
| `ell=5` | `log(2)/30` | `1/2` |

La scala `1/[L_b(L_b+1)]` compensa, tra compiti diversi, il decadimento più
rapido dei ranghi fini. L'inversione formale dei coefficienti del gruppo
amplificherebbe un errore additivo al massimo di `exp[kappa]` per quel tratto.
È un controllo matematico dell'attenuazione locale, non una dimostrazione
dell'esistenza di un inverso CPTP universale o della convergenza del training.
La scelta `kappa = log(2)` è illustrativa, non selezionata sperimentalmente.

**Condizione decisiva sugli ingressi.** La frazione conservata è riferita
allo stato immediatamente precedente a quella diffusione. Se un rango fine
è già stato fortemente attenuato prima, applicare dopo una diffusione più
debole non lo ripristina. Per sfruttare la proposta, i bersagli del compito
fine devono provenire da livelli di rumore adeguati: per esempio essere
costruiti dal bersaglio di training con la forza debole prevista, oppure
essere gli stati vicini ai dati di una traiettoria coerente. Una piccola
intensità nell'ultimo tratto, da sola, non garantisce una componente finale
ancora apprezzabile rispetto al dato originale.

**Composizione e generazione.** Se tutti i tratti sono applicati in sequenza
allo stesso stato, l'attenuazione finale resta
`exp[-ell(ell+1) sum_b a_b]`: il buon condizionamento locale non implica il
buon condizionamento dell'inverso complessivo. Occorre definire quali gruppi
siano fasi di ottimizzazione e quali corrispondano a passi della catena,
collegare coerentemente ingressi e uscite e valutare la catena intera da
`I/d`. Il decadimento relativo tra ranghi all'interno di un singolo tratto
resta quello del calore standard; quello che cambia è la forza usata nei
diversi compiti. Conservare nella loss le bande precedenti e i controlli
globali. Se il processo diretto complessivo è reso più debole, misurare
anche la distanza del suo stato terminale dal prior `I/d`.

**Verifiche e criteri di accettazione da formalizzare.** Confrontare forza
comune e forza adattata per gruppo, con e senza curriculum, mantenendo
architettura, dati, inizializzazioni e budget dichiarati. Verificare sia la
frazione di segnale conservata rispetto al bersaglio originale sia il rapporto
tra coefficienti prima e dopo ogni tratto. Misurare errori complessi per rango,
metriche globali e di lettura, conservazione delle bande già apprese e costi
totali. Il criterio finale resta il miglioramento della generazione composta
su osservazioni indipendenti, non soltanto dei singoli compiti su ingressi
esatti. Se la supervisione rimane esclusivamente finale da `I/d`, cambiare
solo la traiettoria forward non modifica quella loss: il protocollo deve
esplicitare dove entrano i bersagli a diffusione adattata.

**Riferimenti:** [legge di decadimento e prior](../materials/paper/sections/theory.tex),
[generatore corrente](../src/spin_quantum_diffusion/quantum/single_spin.py)
e [limiti della supervisione su ingressi esatti](TWO_SPIN_SUPERVISION_REPORT.md#validation-selection).
La relazione quantitativa proposta è una deduzione dalla legge del modello,
non una nuova evidenza sperimentale.

**Esiti e decisioni:** l'attenuazione del singolo compito può essere limitata
con la scala indicata. Il beneficio nell'apprendimento e nella generazione
composta resta da verificare; nessuna esecuzione, selezione o conferma.
Il protocollo e le priorità restano nel piano autorevole.

### 9. Capacità limitata delle ancille di rimuovere entropia

Ogni collisione unitaria con un'ancilla pura di un qubit, inizialmente
indipendente e poi scartata, può ridurre l'entropia di von Neumann del registro
al massimo di un bit. Partendo da `I/d`, dopo `K` collisioni di questo tipo
vale quindi `S(rho_out) >= max(0, log2(d) - K)`, con entropia espressa in bit.
Il limite segue dalla conservazione dell'entropia complessiva sotto unitarie
e dalla subadditività: l'ancilla scartata può avere al massimo un bit di
entropia.

Aumentando spin o numero di siti senza aumentare le risorse dissipative,
alcuni bersagli potrebbero diventare irraggiungibili anche con ottimizzazione
perfetta. È una deduzione teorica per l'architettura descritta; non è ancora
stato verificato se questo vincolo spieghi gli errori osservati nei bersagli
del progetto. Il limite riguarda il canale incondizionato, senza
postselezione degli esiti.

**Riferimenti:** [architettura delle collisioni](../materials/paper/sections/theory.tex),
[protocollo della griglia](protocols/SPIN_GRID_V1.md) e
[Lieb, proprietà dell'entropia quantistica](https://web.math.princeton.edu/~lieb/cm-summerschool.pdf).

- **Soluzioni proposte da considerare:** da aggiungere.
- **Verifiche e criteri di accettazione:** da definire per ciascuna proposta.
- **Esiti e decisioni:** da compilare dopo la valutazione.

### 10. Crescita esponenziale del costo di simulazione

Con `N` spin uguali, la dimensione del registro è `(2j+1)^N` e la matrice
densità contiene `(2j+1)^(2N)` elementi. Aumentare insieme il numero di
variabili e la risoluzione locale rende rapidamente costosi memoria,
calcolo dei gradienti e valutazione degli stati nella simulazione densa.

Questa crescita è un vincolo della rappresentazione numerica attuale. Non
costituisce, da sola, una dimostrazione di vantaggio quantistico o un limite
inferiore per ogni possibile simulatore classico strutturato.

**Riferimento:** [teoria dell'estensione a più spin](../materials/paper/sections/theory.tex).

- **Soluzioni proposte da considerare:** da aggiungere.
- **Verifiche e criteri di accettazione:** da definire per ciascuna proposta.
- **Esiti e decisioni:** da compilare dopo la valutazione.

### 11. Possibile peggioramento dell'addestrabilità

Circuiti più grandi o profondi possono presentare gradienti molto piccoli
e richiedere molte più valutazioni. I barren plateau costituiscono un rischio
teorico noto per determinate famiglie di circuiti e condizioni di
inizializzazione, ma non sono una causa già dimostrata delle difficoltà
osservate nel progetto.

L'aumento della capacità del circuito non garantisce quindi, da solo, un
apprendimento migliore. L'incidenza di questo rischio nell'architettura
SpinDiffusion deve essere verificata direttamente.

**Riferimenti:** [McClean et al., barren plateau](https://arxiv.org/abs/1803.11173)
e [diagnosi dell'addestramento e limiti interpretativi](DIAGNOSTIC_REPORT.md).

- **Soluzioni proposte da considerare:** da aggiungere.
- **Verifiche e criteri di accettazione:** da definire per ciascuna proposta.
- **Esiti e decisioni:** da compilare dopo la valutazione.

### 12. Trasferimento difficile dalla simulazione all'hardware

In simulazione sono disponibili matrici e probabilità esatte. Su un
dispositivo bisogna stimare le quantità richieste tramite misure e
contabilizzare ripetizioni, rumore, preparazione e reset delle ancille,
profondità dei circuiti e implementazione della misura finale.

Il costo dell'addestramento e della verifica potrebbe diventare maggiore
di quello della generazione dei campioni. Questo rischio non è ancora
quantificato per il progetto: le verifiche numeriche di fisicità e i piloti
con POVM finita non equivalgono a una validazione hardware o a una misura
di efficienza operativa.

**Riferimenti:** [discussione dei limiti hardware](../materials/paper/sections/discussion.tex)
e [condizioni del piano per il passaggio all'hardware](SPINDIFFUSION_PAPER_PLAN.md#scientific-safeguards-and-out-of-scope-gates).

- **Soluzioni proposte da considerare:** da aggiungere.
- **Verifiche e criteri di accettazione:** da definire per ciascuna proposta.
- **Esiti e decisioni:** da compilare dopo la valutazione.
