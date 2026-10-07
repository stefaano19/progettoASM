# progettoASM

**Studio di un Sistema Agentico in Reti Sociali — Co-evoluzione tra Topologia, Cognizione LLM e Interventi di Fact-Checking**

Progetto accademico per il corso di *Analisi di Social Network e Media*. Il framework simula come opinioni testuali generate da agenti LLM, topologia della rete sociale e interventi correttivi (fact-checking) si influenzino a vicenda in un ciclo chiuso di co-evoluzione, anziché come fasi sequenziali indipendenti.

Repository: [`github.com/stefaano19/progettoASM`](https://github.com/stefaano19/progettoASM)

---

## Indice

- [Panoramica](#panoramica)
- [Architettura: il Loop di Co-evoluzione](#architettura-il-loop-di-co-evoluzione)
- [Stack Tecnologico](#stack-tecnologico)
- [Struttura del Progetto](#struttura-del-progetto)
- [Installazione](#installazione)
- [Configurazione](#configurazione)
- [Esecuzione](#esecuzione)
- [Le Quattro Fasi](#le-quattro-fasi)
- [Metriche](#metriche)
- [Risultati](#risultati)
- [Limitazioni](#limitazioni)
- [Riferimenti](#riferimenti)

---

## Panoramica

Il problema centrale affrontato dal progetto è la modellazione di come:

- gli **stati interni degli agenti** (opinioni testuali e cognitive),
- la **topologia della rete** (legami sociali),
- e gli **interventi esterni** (fact-checking)

si influenzino reciprocamente e simultaneamente, in un ciclo chiuso di interazione continua anziché in pipeline sequenziali e indipendenti.

Il dataset di partenza è **`ogbl-collab`** (Open Graph Benchmark), una rete di co-autoraggio accademico: la prossimità nello spazio vettoriale degli agenti determina la probabilità di formazione o recisione dei legami sociali.

## Architettura: il Loop di Co-evoluzione

Il sistema è organizzato su tre livelli che interagiscono in un ciclo chiuso:

| Livello | Componenti | Ruolo |
|---|---|---|
| **Cognitivo** | Agenti generativi (LLM) | Generano opinioni testuali e stato cognitivo |
| **Strutturale** | GNN (GraphSAGE) + grafo dinamico | Calcolano il rewiring della topologia (link prediction) |
| **Intervento** | Linear Threshold + CELF | Gestiscono la diffusione del contagio e l'iniezione dei fact-checker |

A ogni step: gli agenti attivi leggono il feed dei vicini e decidono tramite LLM → le transizioni di stato perturbano gli embedding → la GNN viene addestrata e calcola gli score dei link → il rewiring aggiorna la topologia $G_{t+1}$, che determina il feed e il vicinato dello step successivo. In Fase 3 CELF sceglie i nodi da convertire in fact-checker.

## Stack Tecnologico

| Componente | Tecnologia | Motivazione |
|---|---|---|
| Linguaggio | Python ≥ 3.10 | Ecosistema Data Science |
| Graph Neural Network | **GraphSAGE** (2 layer, aggregazione media, PyTorch con matrici sparse) | Apprendimento induttivo su un grafo che cambia a ogni step |
| Inferenza LLM | **vLLM** (server locale OpenAI-compatible) | Continuous batching, nessun rate limit, dati in locale |
| Modello LLM | `casperhansen/llama-3-8b-instruct-awq` | Quantizzato AWQ, compatibile con GPU T4 (15 GB VRAM) |
| Modello di diffusione | Linear Threshold esteso, modulato dall'LLM | Dinamica di contagio dell'opinione |
| Influence Maximization | Greedy su obiettivo a soglia / **CELF** su Independent Cascade | Selezione dei seed fact-checker |
| Dataset | `ogbl-collab` (Open Graph Benchmark) | Rete di co-autoraggio accademico, 235.868 nodi |

## Struttura del Progetto

```
progettoASM/
├── config.yaml                    # Configurazione centrale (allineata al notebook)
├── requirements.txt
├── notebooks/
│   └── kaggle_full_run.ipynb      # Notebook che esegue l'intera pipeline
├── src/
│   ├── orchestrator.py            # SimulationOrchestrator — ciclo agenti → GNN → rewiring → metriche
│   ├── graph/
│   │   ├── data_loader.py         # Download/caching di ogbl-collab
│   │   ├── extractor.py           # Campionamento del sottografo (Forest Fire, BFS, random walk, random)
│   │   ├── community.py           # Community detection (Louvain, Label Propagation)
│   │   ├── metrics.py             # Centralità, modularità, ECI, metriche di opinione
│   │   └── network_manager.py     # Grafo dinamico: stati, post, embedding, rewiring
│   ├── agents/
│   │   ├── agent.py               # Agente cognitivo (percezione → LLM → azione → transizione)
│   │   ├── llm_client.py          # Client LLM (vLLM/Ollama/Gemini), retry, timeout, cache su disco
│   │   ├── prompts.py             # Prompt di sistema/utente, etichette di influenza
│   │   ├── seeder.py              # Selezione dei "pazienti zero"
│   │   └── state_machine.py       # Transizioni S/I/R/F (Linear Threshold modulato)
│   ├── gnn/
│   │   ├── embeddings.py          # Gestione degli embedding iniziali dei nodi
│   │   ├── model.py               # GraphSAGEModel (PyTorch; NumPy solo per prove senza training)
│   │   ├── trainer.py             # GNNTrainer — training e link prediction
│   │   └── rewirer.py             # Rewiring dagli score GNN (swap a densità costante)
│   ├── influence/
│   │   ├── celf.py                # Selezione dei seed fact-checker
│   │   ├── injector.py            # FactCheckerInjector
│   │   └── metrics.py             # Reach e copertura efficace dei fact-checker, delta
│   └── utils/
│       ├── logger.py              # SimLogger (log JSONL)
│       ├── checkpoint.py          # CheckpointManager (resume tra sessioni)
│       ├── config.py              # Caricamento di config.yaml in dataclass
│       └── seed.py                # Riproducibilità (set_all_seeds)
├── relazione/                     # Relazione tecnica (LaTeX)
└── results/                       # Generata a runtime: metriche, report, figure, log, checkpoint
```

## Installazione

Il progetto è pensato per essere eseguito su **Kaggle** (2 × GPU T4) tramite `notebooks/kaggle_full_run.ipynb`, che clona il repository, installa le dipendenze e avvia vLLM.

Requisiti:
- Un token Hugging Face (`HF_TOKEN`), salvato come **Kaggle Secret**, per scaricare il modello.
- GPU con almeno 15 GB di VRAM per il modello AWQ (es. NVIDIA T4).

In locale (con GPU CUDA):

```bash
git clone https://github.com/stefaano19/progettoASM.git
cd progettoASM
pip install -r requirements.txt
export HF_TOKEN="il-tuo-token"
```

## Configurazione

I parametri di sessione si impostano in testa al notebook:

| Parametro | Descrizione | Esempio |
|---|---|---|
| `USE_MOCK_LLM` | `False` = LLM reale (vLLM); `True` = mock per prove rapide | `False` |
| `PHASE2_STEPS` | **Nuovi** step di Fase 2 da eseguire in questa sessione | `50` |
| `PHASE3_STEPS` | Step di Fase 3 (`0` nelle sessioni di sola Fase 2) | `30` |
| `CELF_BUDGET_K` | Numero di fact-checker da iniettare | `20` |
| `CONTROL_RUN` | `True` = Fase 3 di controllo, stessi step senza fact-checker | `False` |
| `FORCE_INJECTION` | Inietta anche se l'infection rate è sotto `influence.activation_threshold` | `True` |
| `SAMPLING_STRATEGY` | Campionamento del sottografo | `forest_fire` |
| `FOREST_FIRE_PROB` | Forward probability del Forest Fire (0.4–0.7) | `0.5` |
| `TARGET_NODES` | Nodi del sottografo | `5000` |
| `RESUME_FROM_CKPT` | Riprende da un checkpoint (anche da una versione precedente del notebook) | `True` |

Parametri principali di `config.yaml`:

| Parametro | Descrizione | Valore |
|---|---|---|
| `simulation.activation_probability` | Frazione di nodi attivati a ogni step | `0.15` |
| `simulation.initial_infection_rate` | Frazione di pazienti zero | `0.15` |
| `simulation.memory_window` | Post per vicino nel feed | `3` |
| `simulation.rewiring_cooldown` | Rewiring ogni N step | `2` |
| `simulation.relapse_threshold` / `relapse_min_susceptibility` | Ricaduta R → I | `0.6` / `0.8` |
| `simulation.max_llm_failure_rate` | Lo step si ferma se eccezioni + fallback LLM superano questa quota | `0.5` |
| `simulation.smart_cache` | Riuso della risposta LLM a contesto invariato (congela gli agenti) | `false` |
| `llm.local.timeout` | Timeout per richiesta (secondi), poi retry e fallback | `300` |
| `gnn.rewire_mode` | `swap`: ogni arco aggiunto sostituisce quello con score più basso | `swap` |
| `gnn.max_new_edges_per_step` | Archi sostituiti per step di rewiring | `50` |
| `gnn.rewire_threshold_add` / `remove` | Soglie degli score per aggiunta/rimozione | `0.65` / `0.35` |
| `influence.celf_objective` | `threshold` (allineato alla state machine) o `ic` | `threshold` |
| `influence.fc_protection_threshold` | S → R se la frazione di vicini F supera la soglia | `0.10` |
| `influence.fc_resistance_threshold` | I → R se la frazione di vicini F supera la soglia | `0.25` |

## Esecuzione

Il notebook esegue le quattro fasi in sequenza; checkpoint (`.pkl`) e metriche (`results/metrics_history.csv`) vengono salvati a ogni step, così la run può essere ripresa nella sessione Kaggle successiva (limite di 12 ore).

Piano delle sessioni:

| Sessione | Parametri | Contenuto |
|---|---|---|
| 1 | `PHASE2_STEPS=50`, `PHASE3_STEPS=0` | Fase 0 + step 0–49 |
| 2 | resume, `PHASE2_STEPS=50`, `PHASE3_STEPS=0` | step 50–99 |
| 3 | resume dal checkpoint finale di Fase 2, `PHASE2_STEPS=0`, `PHASE3_STEPS=30` | Fase 3 con fact-checker |
| 4 | stesso checkpoint della sessione 3, `CONTROL_RUN=True` | Fase 3 di controllo |

La durata di uno step con l'LLM reale va misurata nella prima sessione: se 50 step non entrano nelle 12 ore, ridurre `PHASE2_STEPS` e aggiungere sessioni.

Prima di lanciare una sessione lunga conviene eseguire 3–4 step e controllare nella tabella stampata dal notebook:
- la colonna **FB** (risposte di fallback dell'LLM): deve essere 0 o quasi;
- l'intervallo **Score min-max** del link predictor: deve coprire buona parte di (0, 1);
- il prompt di un agente che ha appena cambiato stato (`orch._agents[n].system_prompt`): stato e influenza devono essere corretti.

L'effetto dell'intervento è la differenza tra la run con fact-checker (sessione 3) e quella di controllo (sessione 4). Lo calcola l'ultima cella del notebook, **Confronto Intervento vs Controllo**: va eseguita nella sessione 4 dopo aver aggiunto come input l'output della sessione 3. Controlla che le due run partano dallo stesso stato, stampa l'effetto finale e quello per step, e salva `results/phase3_comparison.json` e `results/figures/phase3_comparison.png`.

## Le Quattro Fasi

### Fase 0 — Dati e baseline
Avvia vLLM, scarica `ogbl-collab`, estrae un sottografo di 5.000 nodi con **Forest Fire Sampling**, valida il campione contro il grafo originale (distribuzione dei gradi, clustering, modularità), rileva le community con Label Propagation e calcola le metriche di baseline.

### Fase 1 — Agenti LLM
Crea un agente per nodo. Il prompt di sistema contiene la personalità (dalla community), l'influenza nella rete (dai percentili del grado: top 10% "high", dal 60° al 90° percentile "medium") e lo stato corrente; viene ricostruito a ogni cambio di stato. Il 15% dei nodi più centrali parte infetto (pazienti zero).

### Fase 2 — Co-evoluzione
A ogni step il `SimulationOrchestrator` attiva il 15% dei nodi. Ogni agente attivo legge il feed dei vicini (ordinato per step, con ordine casuale riproducibile a parità di step), chiama l'LLM e pubblica un post etichettato con il suo nuovo stato. La state machine decide la transizione combinando la pressione dei vicini con la suscettibilità e lo stato proposto dall'LLM. Le transizioni spostano gli embedding lungo una direzione comune (I in un verso, R e F nel verso opposto); la GNN viene addestrata sugli archi correnti e i suoi score guidano il rewiring a densità costante.

Se l'LLM non risponde (eccezioni o risposte di fallback oltre `max_llm_failure_rate`), lo step si interrompe prima di modificare il grafo e la run si riprende dall'ultimo checkpoint.

### Fase 3 — Intervento e fact-checking
CELF sceglie `CELF_BUDGET_K` nodi suscettibili e li converte in fact-checker (stato F). Con l'obiettivo `threshold` la selezione massimizza i nodi su cui la pressione dei fact-checker supera le soglie che producono davvero una transizione (S → R e I → R). L'effetto si misura confrontando due run di Fase 3 partite dallo stesso checkpoint, una con fact-checker e una di controllo; ogni richiesta all'LLM usa un seed derivato dal prompt, così le due run restano confrontabili.

## Metriche

- **Echo Chamber Index (ECI)** e **modularità**: quanto gli archi restano dentro le community della Fase 0. Sono strutturali e non dipendono dalle opinioni.
- **Opinion homophily** e **belief assortativity**: quanto i nodi con la stessa opinione sono connessi tra loro. Codifica: S = 0, I = +1, R = F = −1.
- **Belief polarisation**: varianza delle opinioni normalizzata al massimo teorico. Cresce ogni volta che un nodo neutrale (S) passa a una posizione netta, quindi anche quando i fact-checker convertono S in R.
- **Reach dei fact-checker**: nodi entro `reach_hops` salti da un F; **copertura efficace**: nodi su cui la pressione F supera le soglie di transizione.
- **Diagnostica per step** (log JSONL): chiamate LLM, risposte di fallback, distribuzione degli score del link predictor.

## Risultati

Run documentata: sottografo di 5.000 nodi, Fase 2 di 100 step (0–99), due run di Fase 3 di 30 step (100–129) partite dallo stesso checkpoint, con LLM reale (`casperhansen/llama-3-8b-instruct-awq`).

### Fase 2 — Co-evoluzione

| Step | S | I | R | ECI | Modularity Q | Opinion homophily | Belief assortativity |
|---|---|---|---|---|---|---|---|
| 0 | 3965 | 841 | 194 | 0.832 | 0.828 | 0.629 | 0.268 |
| 49 | 1070 | 2022 | 1908 | 0.850 | 0.854 | 0.759 | 0.541 |
| 99 | 978 | 2032 | 1990 | 0.874 | 0.877 | 0.814 | 0.659 |

- Il contagio satura intorno al 41% dei nodi (picco di 2034 infetti allo step 88); i resistenti crescono con lo stesso andamento e restano poco sotto gli infetti.
- La segregazione cresce su entrambi i fronti, ma molto più sulle opinioni (belief assortativity da 0.27 a 0.66) che sulla struttura (ECI da 0.83 a 0.87).
- Il rewiring sostituisce 2500 archi (circa il 10% della rete) a densità costante; a fine Fase 2 la rete ha 86 componenti connesse.

### Fase 3 — Intervento vs controllo (step 129)

CELF (obiettivo `threshold`, `CELF_BUDGET_K=20`) ha scelto 20 suscettibili con una stima di 162 nodi coperti. L'infection rate di partenza (0.406) superava la soglia di attivazione, quindi l'iniezione non è stata forzata.

| Metrica | Con fact-checker | Controllo | Effetto |
|---|---|---|---|
| Nodi S | 798 | 974 | −176 (di cui 20 seed divenuti F) |
| Nodi I | 2020 | 2024 | −4 |
| Nodi R | 2162 | 2002 | +160 |
| Infection rate | 0.4040 | 0.4048 | −0.0008 |
| Echo Chamber Index | 0.8855 | 0.8858 | −0.0003 |
| Modularity Q | 0.8831 | 0.8835 | −0.0004 |
| Opinion homophily | 0.8305 | 0.8306 | −0.0001 |
| Belief assortativity | 0.7042 | 0.6922 | +0.0119 |
| Belief polarisation | 0.8394 | 0.8052 | +0.0342 |
| Transizioni in 30 step | 177 | 17 | +160 |

- **Protezione dei suscettibili**: circa 156 suscettibili in più diventano resistenti, in linea con la stima di CELF. L'effetto si esaurisce presto: 135 delle 177 transizioni avvengono nei primi 10 step e a fine run la copertura efficace è di 5 nodi.
- **Infetti**: effetto trascurabile (−4 nodi). Un infetto si converte solo se almeno il 25% dei vicini è fact-checker, condizione che 20 seed su 5000 nodi non raggiungono quasi mai.
- **Echo chamber e struttura**: ECI, modularità e homophily sono uguali nelle due run; l'intervento agisce sugli stati, non sulla topologia. Assortativity e polarizzazione crescono leggermente perché nodi neutrali (S) passano a una posizione netta (R).

### File


| File | Contenuto |
|---|---|
| `results/metrics_history.csv` | Metriche per step: Fase 2 + Fase 3 con fact-checker |
| `results/phase3_report.json` | Report finale della run con fact-checker (seed CELF, reach, copertura, delta) |
| `results/phase3_report_control.json` | Report finale della run di controllo |
| `results/metrics_history_control.csv` | Metriche per step: Fase 2 + Fase 3 di controllo |
| `results/pipeline_summary*.json` | Riepilogo della pipeline |
| `results/phase3_comparison.json` | Effetto dell'intervento (intervento − controllo), finale e per step |
| `results/figures/` | Validazione del sottografo, evoluzione di Fase 2, confronto di Fase 3, rete iniziale e finale |
| `results/logs/` | Log JSONL con metriche, transizioni e rewiring per step |

## Limitazioni

- **Una sola realizzazione.** Il costo per step consente una sola run per configurazione: effetti piccoli andrebbero confermati con più seed.
- **Frammentazione.** Il rewiring non isola singoli nodi, ma può staccare piccoli gruppi dalla rete.
- **Ripetibilità dell'LLM.** Il seed per richiesta rende ripetibili le risposte a parità di prompt, ma vLLM non garantisce il determinismo completo con batching variabile.
- **Obiettivo di selezione miope.** L'obiettivo `threshold` considera l'effetto immediato dei fact-checker sui vicini, non la dinamica futura.
- **Community statiche.** ECI e modularità usano le community della Fase 0.
- **Personalità.** La personalità dell'agente dipende dall'identificativo della community (modulo 4), non da caratteristiche della community.
- **Scala del modello.** Llama 3 8B quantizzato AWQ è un compromesso dettato dalla GPU disponibile.

## Riferimenti

- Dataset: [Open Graph Benchmark — `ogbl-collab`](https://ogb.stanford.edu/docs/linkprop/#ogbl-collab)
- Hamilton et al., *Inductive Representation Learning on Large Graphs* (GraphSAGE)
- Kempe, Kleinberg, Tardos, *Maximizing the Spread of Influence through a Social Network*
- Leskovec et al., *Cost-effective Outbreak Detection in Networks* (CELF)
- Serving LLM: [vLLM](https://github.com/vllm-project/vllm)
