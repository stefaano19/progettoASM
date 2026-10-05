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
- [Esecuzione della Pipeline](#esecuzione-della-pipeline)
- [Le Quattro Fasi](#le-quattro-fasi)
- [Risultati Principali](#risultati-principali)
- [Limitazioni e Sviluppi Futuri](#limitazioni-e-sviluppi-futuri)
- [Riferimenti](#riferimenti)

---

## Panoramica

Il problema centrale affrontato dal progetto è la modellazione realistica di come:

- gli **stati interni degli agenti** (opinioni testuali e cognitive),
- la **topologia della rete** (legami sociali),
- e gli **interventi esterni** (fact-checking)

si influenzino reciprocamente e simultaneamente, in un ciclo chiuso di interazione continua anziché in pipeline sequenziali e indipendenti.

Il dataset di partenza è **`ogbl-collab`** (Open Graph Benchmark), una rete di co-autoraggio accademico, riutilizzata qui in modo innovativo: la prossimità nello spazio vettoriale cognitivo degli agenti determina la probabilità di formazione o recisione dei legami sociali.

**Target di riferimento:** ricercatori in Network Science, studiosi di sistemi multi-agente, data scientist.

## Architettura: il Loop di Co-evoluzione

Il sistema è organizzato su tre livelli che interagiscono in un ciclo chiuso:

| Livello | Componenti | Ruolo |
|---|---|---|
| **Cognitivo** | Agenti Generativi (LLM) | Generano opinioni testuali e stato cognitivo |
| **Strutturale** | GNN (GraphSAGE) + Grafo Dinamico | Calcolano il rewiring della topologia (link prediction) |
| **Intervento** | Linear Threshold + Algoritmo CELF | Gestiscono la diffusione del contagio e l'iniezione dei fact-checker |

Il ciclo si chiude in 6 passaggi: gli agenti generano opinioni che perturbano gli embedding → la GNN calcola il rewiring → il nuovo output testuale alimenta la macchina a stati → la topologia aggiornata $G_{t+1}$ retroagisce sugli agenti → le transizioni innescano il trigger su soglia d'infezione → CELF inietta i nodi Fact-Checker, richiudendo il ciclo.

## Stack Tecnologico

| Componente | Tecnologia | Motivazione |
|---|---|---|
| Linguaggio | Python ≥ 3.10 | Ecosistema Data Science predominante |
| Graph Neural Network | **GraphSAGE** (Sample and Aggregate) | Apprendimento induttivo, necessario per grafi con topologia che muta a ogni step, senza dover riaddestrare da zero |
| Inferenza LLM | **vLLM** (server locale OpenAI-compatible) | Continuous batching, throughput elevato, nessun rate-limit da API cloud, piena sovranità sui dati |
| Modello LLM | `casperhansen/llama-3-8b-instruct-awq` | Quantizzato AWQ, compatibile con GPU T4 (15 GB VRAM) |
| Modello di diffusione | Linear Threshold (LT) esteso, modulato dall'LLM | Dinamica di contagio dell'opinione/stato |
| Influence Maximization | **CELF** / greedy | Selezione dei seed fact-checker; obiettivo allineato alle regole della simulazione (IC Monte Carlo disponibile come opzione) |
| Dataset | `ogbl-collab` (Open Graph Benchmark) | Rete di co-autoraggio accademico: 235.868 nodi, 967.632 archi non orientati (2.358.104 collaborazioni con timestamp prima della deduplicazione) |

## Struttura del Progetto

```
progettoASM/
├── config.yaml                    # Configurazione centrale della pipeline
├── requirements.txt
├── notebooks/
│   └── kaggle_full_run.ipynb      # Notebook orchestratore end-to-end
├── src/
│   ├── orchestrator.py            # SimulationOrchestrator — motore centrale
│   ├── graph/
│   │   ├── data_loader.py         # Download/caching di ogbl-collab
│   │   ├── extractor.py           # Campionamento del sottografo (Forest Fire, BFS, RWR, random)
│   │   ├── community.py           # Community detection (Louvain, Label Propagation)
│   │   ├── metrics.py             # Centralità, modularità, ECI, polarizzazione e assortatività delle opinioni
│   │   └── network_manager.py     # Layer unico di astrazione/persistenza del grafo dinamico
│   ├── agents/
│   │   ├── agent.py               # Agente cognitivo (percezione → cognizione → azione)
│   │   ├── llm_client.py          # Wrapper LLM portabile (vLLM/Ollama/Gemini + cache su disco)
│   │   ├── prompts.py             # Prompt di sistema/utente degli agenti
│   │   ├── seeder.py              # Selezione dei "pazienti zero"
│   │   └── state_machine.py       # Transizioni S/I/R/F (Linear Threshold modulato)
│   ├── gnn/
│   │   ├── embeddings.py          # EmbeddingManager (Word2Vec)
│   │   ├── model.py               # GraphSAGEModel (PyTorch Geometric / NumPy fallback)
│   │   ├── trainer.py             # GNNTrainer — training + link prediction
│   │   └── rewirer.py             # Rewiring dagli score GNN (swap a densità costante, vincoli di sicurezza)
│   ├── influence/
│   │   ├── celf.py                # Selezione seed (obiettivo "threshold" o IC con CELF)
│   │   ├── injector.py            # FactCheckerInjector
│   │   └── metrics.py             # Reach e copertura efficace dei fact-checker, delta
│   └── utils/
│       ├── logger.py              # SimLogger (log JSONL)
│       ├── checkpoint.py          # CheckpointManager (resume cross-sessione)
│       ├── config.py              # Caricamento config.yaml in dataclass
│       └── seed.py                # Riproducibilità (set_all_seeds)
├── phase0_run.py … phase3_run.py   # Entry point delle singole fasi
├── scripts/
│   ├── check_propagation.py
│   └── compare_phase3.py          # Effetto dell'intervento: run con fact-checker vs controllo
├── tests/                         # Test pytest (126 test)
├── CHANGELOG_FIX.md               # Elenco delle correzioni
└── results/                       # Risultati della run finale (vedi sotto); figures/ e checkpoints/
                                   # sono generati a runtime
```

## Installazione

Il progetto è pensato per essere eseguito su **Kaggle** (GPU T4) tramite il notebook `kaggle_full_run.ipynb`, ma è portabile in locale con una GPU CUDA compatibile.

```bash
git clone https://github.com/stefaano19/progettoASM.git
cd progettoASM

pip install -r requirements.txt

# Setup del server LLM locale (vLLM, OpenAI-compatible)
pip install vllm
```

**Requisiti aggiuntivi:**
- Un token Hugging Face (`HF_TOKEN`) per scaricare il modello `casperhansen/llama-3-8b-instruct-awq`.
- Su Kaggle, il token va salvato come **Kaggle Secret**; in locale, come variabile d'ambiente:
  ```bash
  export HF_TOKEN="il-tuo-token"
  ```
- GPU con almeno 15 GB di VRAM per il modello quantizzato AWQ (es. NVIDIA T4).

## Configurazione

I parametri principali si impostano in testa al notebook (o in `config.yaml`):

| Parametro | Descrizione | Esempio |
|---|---|---|
| `USE_MOCK_LLM` | `False` = usa l'LLM reale (vLLM); `True` = mock per debug rapido | `False` |
| `PHASE2_STEPS` | Numero di **nuovi** step da eseguire in Fase 2 in questa sessione (non il totale cumulato) | `50` |
| `PHASE3_STEPS` | Step di simulazione in Fase 3 (`0` nelle sessioni intermedie di Fase 2) | `30` |
| `CELF_BUDGET_K` | Numero di fact-checker da iniettare | `20` |
| `CONTROL_RUN` | `True` = Fase 3 di controllo, stessi step ma senza fact-checker | `False` |
| `FORCE_INJECTION` | Inietta anche se l'infection rate è sotto `influence.activation_threshold` (segnalato nel report) | `True` |
| `SAMPLING_STRATEGY` | Strategia di campionamento del sottografo | `forest_fire` \| `bfs_seed` \| `random_walk` \| `random_nodes` |
| `FOREST_FIRE_PROB` | Forward probability del Forest Fire Sampling (0.4–0.7) | `0.5` |
| `TARGET_NODES` | Dimensione del sottografo campionato | `5000` |
| `RESUME_FROM_CKPT` | Riprendi da un checkpoint salvato (cross-sessione, utile su Kaggle) | `True` |

Parametri principali di `config.yaml` introdotti con le correzioni:

| Parametro | Descrizione | Default |
|---|---|---|
| `gnn.rewire_mode` | `swap`: ogni arco aggiunto sostituisce l'arco esistente con lo score più basso (densità costante); `threshold`: rimozioni solo sotto soglia | `swap` |
| `gnn.max_new_edges_per_step` | Archi sostituiti per step | `50` |
| `gnn.rewire_candidate_sources` | Nodi sorgente esplorati a ogni step per i candidati (diversi a ogni step) | `500` |
| `influence.celf_objective` | `threshold`: obiettivo allineato alla state machine; `ic`: Independent Cascade Monte Carlo | `threshold` |
| `influence.fc_protection_threshold` | S → R se la frazione di vicini F supera la soglia | `0.10` |
| `influence.fc_resistance_threshold` | I → R se la frazione di vicini F supera la soglia | `0.25` |
| `influence.reach_hops` | Raggio della metrica di reach dei fact-checker | `1` |

## Esecuzione della Pipeline

L'intera pipeline è orchestrata dal notebook `kaggle_full_run.ipynb`, suddiviso in 4 fasi eseguite in sequenza nella stessa sessione (o riprese via checkpoint):

```
Fase 0 → Fase 1 → Fase 2 → Fase 3
```

Ogni fase salva automaticamente checkpoint (`.pkl`) e metriche (`metrics_history.csv`), così l'esecuzione può essere interrotta e ripresa — utile per superare i limiti di tempo delle sessioni Kaggle gratuite.

Con l'LLM reale uno step dura circa 6 minuti con vLLM su due T4 (`VLLM_TENSOR_PARALLEL = 2`), circa 12 su una sola. Piano delle sessioni Kaggle usato per i risultati (limite 12 ore per sessione), con Fase 2 da 100 step:

| Sessione | Parametri | Contenuto |
|---|---|---|
| 1 | `PHASE2_STEPS=50`, `PHASE3_STEPS=0` | Fase 0 + step 0–49 |
| 2 | resume, `PHASE2_STEPS=50`, `PHASE3_STEPS=0` | step 50–99 |
| 3 | resume dal checkpoint finale, `PHASE2_STEPS=0`, `PHASE3_STEPS=30` | Fase 3 con fact-checker (step 100–129) |
| 4 | stesso checkpoint della sessione 3, `CONTROL_RUN=True` | Fase 3 di controllo (step 100–129) |

Poi:

```bash
python scripts/compare_phase3.py \
    --treatment results/phase3_report.json --control results/phase3_report_control.json \
    --csv results/metrics_history.csv --csv-control results/metrics_history_control.csv
```

## Le Quattro Fasi

### Fase 0 — Setup, Data Ingestion e Baseline
Inizializza il server vLLM, scarica/carica `ogbl-collab`, estrae un sottografo di 5.000 nodi tramite **Forest Fire Sampling**, rileva le community (Label Propagation) e calcola le metriche baseline (centralità, modularità, Echo Chamber Index).

- **Validazione strutturale:** il sottografo preserva la distribuzione dei gradi a legge di potenza e le proprietà strutturali chiave del grafo originale (232.865 nodi → 5.000 nodi, rapporto di campionamento ~2.1%).
- **Baseline:** 415 community, Modularity Q = 0.8265, Echo Chamber Index = 0.8318.

### Fase 1 — Livello Cognitivo e Agenti LLM
Instanzia gli agenti (`agent.py`), il client LLM (`llm_client.py`), seleziona i "pazienti zero" (`seeder.py`, 15% della rete) e attiva la macchina a stati (`state_machine.py`) che governa le transizioni S → I → R → F secondo un modello Linear Threshold modulato dalla suscettibilità cognitiva valutata dall'LLM.

### Fase 2 — Dinamiche di Rete e Co-evoluzione
Il `SimulationOrchestrator` esegue il ciclo ricorsivo: **ciclo Agenti** (chiamate LLM parallele, batched da vLLM) → **ciclo GNN** (training + link prediction) → **ciclo di Rewiring**. Le transizioni di stato perturbano gli embedding lungo una direzione "ideologica" comune (I in un verso, R e F nel verso opposto), quindi la link prediction riflette la vicinanza di opinione. Il rewiring è a densità costante: ogni arco aggiunto tra nodi con score alto sostituisce l'arco esistente con lo score più basso. Checkpoint frequenti garantiscono resilienza cross-sessione; le soglie dei nodi dipendono solo da seed e id del nodo, quindi restano identiche dopo ogni ripresa.

### Fase 3 — Intervento e Fact-Checking (CELF)
Seleziona i nodi seed e li converte in Fact-Checker (stato F). Con l'obiettivo di default (`threshold`) la selezione massimizza il numero di nodi su cui la pressione dei fact-checker supera le soglie che nella simulazione producono davvero una transizione (S → R e I → R); il calcolo è deterministico ed esatto, con selezione greedy. L'obiettivo `ic` (Independent Cascade + CELF lazy-greedy) resta disponibile.

L'effetto dell'intervento si misura confrontando due run di Fase 3 che partono dallo stesso checkpoint: una con i fact-checker e una di controllo senza (`CONTROL_RUN=True`). Il semplice "prima/dopo" di una run sola mescola l'effetto dei fact-checker con l'evoluzione spontanea della rete. Per rendere le run confrontabili, ogni richiesta all'LLM usa un seed derivato dal prompt.

### Metriche
- **Echo Chamber Index (ECI)** e **modularità**: misurano quanto gli archi restano dentro le community della Fase 0. Sono metriche strutturali: non dipendono dalle opinioni.
- **Opinion homophily** e **belief assortativity**: misurano se i nodi con la stessa opinione sono connessi tra loro (echo chamber ideologiche). Codifica: S = 0, I = +1, R = F = −1.
- **Belief polarisation**: varianza delle opinioni, normalizzata al massimo teorico.
- **Reach dei fact-checker**: nodi entro `reach_hops` salti da un F; **copertura efficace**: nodi su cui la pressione F supera le soglie di transizione.

## Risultati Principali

### Baseline (Fase 0)

La Fase 0 non è toccata dalle correzioni, quindi questi valori restano validi.

| Metrica | Valore |
|---|---|
| Grafo completo (LCC) | 232.865 nodi / 961.883 archi |
| Sottografo (Forest Fire, p = 0.5) | 5.000 nodi / 26.246 archi |
| Densità | 0.0021 |
| Grado medio | 10.50 (originale 8.3) |
| Clustering medio | 0.6988 (originale 0.7204) |
| Community (Label Propagation) | 415 |
| Modularity Q | 0.8265 |
| Echo Chamber Index | 0.8318 |

### Esecuzione

Run del 4–5 ottobre 2026: LLM reale (Llama 3 8B Instruct AWQ su vLLM, 2 × T4), Fase 2 di 100 step (0–99) in due sessioni con ripresa da checkpoint, poi due run di Fase 3 da 30 step (100–129) partite dallo stesso checkpoint: una con 20 fact-checker e una di controllo senza. I dati completi sono in `results/` (`metrics_history.csv`, `metrics_history_control.csv`, `phase3_report.json`, `phase3_report_control.json`, `phase3_comparison.json`).

### Fase 2 — Co-evoluzione (step 0–99)

| Step | S | I | R | ECI | Modularità | Opinion homophily | Belief assortativity |
|---|---|---|---|---|---|---|---|
| 0 | 3964 | 835 | 201 | 0.831 | 0.826 | 0.625 | 0.271 |
| 25 | 1420 | 1840 | 1740 | 0.825 | 0.818 | 0.668 | 0.399 |
| 49 | 1044 | 2039 | 1917 | 0.814 | 0.810 | 0.725 | 0.474 |
| 75 | 930 | 2048 | 2022 | 0.806 | 0.798 | 0.745 | 0.517 |
| 99 | 897 | 2045 | 2058 | 0.801 | 0.788 | 0.755 | 0.541 |

- **Diffusione:** l'infezione cresce rapidamente nei primi 50 step e si stabilizza intorno al 41% dei nodi (picco di 2056 infetti allo step 56). I resistenti crescono più a lungo e alla fine superano gli infetti.
- **Echo chamber di opinione:** la belief assortativity raddoppia (da 0.27 a 0.54) e l'opinion homophily sale da 0.63 a 0.76: i nodi tendono sempre più a essere connessi a chi la pensa come loro.
- **Struttura per community:** ECI e modularità calano leggermente (da 0.831 a 0.801 e da 0.826 a 0.788). Le echo chamber che emergono non coincidono con le community iniziali della rete di co-autoraggio: si formano lungo le opinioni.
- **Rewiring:** 2.500 archi sostituiti in 100 step (circa il 10% della rete) a densità costante. Alla fine la rete ha 46 componenti connesse invece di una: il rewiring non isola singoli nodi, ma può staccare piccoli gruppi.

### Fase 3 — Intervento con fact-checker vs controllo (step 100–129)

L'intervento è partito con un infection rate di 0.409, sopra la soglia di attivazione (0.4), quindi senza forzature. CELF (obiettivo `threshold`) ha scelto 20 nodi suscettibili, con una stima di 163 nodi coperti.

| Metrica (step 129) | Con fact-checker | Controllo | Effetto |
|---|---|---|---|
| Nodi S | 699 | 855 | −156 (di cui 20 sono i seed diventati F) |
| Nodi I | 2040 | 2056 | −16 |
| Nodi R | 2241 | 2089 | +152 |
| Infection rate | 0.408 | 0.411 | −0.003 |
| Belief assortativity | 0.557 | 0.544 | +0.013 |
| Opinion homophily | 0.756 | 0.756 | 0.000 |
| Belief polarisation | 0.858 | 0.829 | +0.029 |
| Echo Chamber Index | 0.791 | 0.792 | −0.001 |
| Modularità | 0.774 | 0.777 | −0.004 |
| Transizioni totali in 30 step | 191 | 59 | +132 |

Andamento nel tempo (S e I, run con fact-checker = FC):

| Step | S (FC) | S (controllo) | I (FC) | I (controllo) |
|---|---|---|---|---|
| 99 | 897 | 897 | 2045 | 2045 |
| 100 | 856 | 896 | 2045 | 2045 |
| 105 | 782 | 891 | 2044 | 2045 |
| 110 | 746 | 883 | 2043 | 2045 |
| 119 | 718 | 875 | 2041 | 2048 |
| 129 | 699 | 855 | 2040 | 2056 |

Con i fact-checker i suscettibili scendono rapidamente nei primi step, mentre nel controllo calano lentamente; gli infetti invece divergono solo col tempo.

- **Protezione dei suscettibili:** l'effetto principale. I fact-checker convertono in resistenti 136 suscettibili in più rispetto al controllo (156 meno i 20 seed), quasi tutti nei primi 10 step (circa 20 per step all'inizio, 2–3 alla fine). La stima di CELF (163 nodi) era nell'ordine di grandezza giusto.
- **Infetti:** l'effetto diretto è quasi nullo, perché un infetto si converte solo se almeno un quarto dei vicini è fact-checker (alla fine solo 9 nodi soddisfano le soglie). Compare però un effetto indiretto che cresce nel tempo: nel controllo gli infetti continuano a salire (da 2045 a 2056), con i fact-checker scendono leggermente (a 2040). I suscettibili protetti non sono più contagiabili, quindi la differenza aumenta step dopo step (da 0 a −16).
- **Echo chamber:** l'intervento non le riduce e anzi aumenta leggermente la belief assortativity (+0.013). I nodi protetti diventano resistenti accanto ad altri resistenti, rafforzando i gruppi di opinione esistenti. Anche la polarizzazione cresce, perché nodi neutrali (S) passano a una posizione netta (R).
- **Struttura:** ECI e modularità sono praticamente uguali nelle due run: l'intervento agisce sulle opinioni, non sulla topologia.

### File dei risultati

| File | Contenuto |
|---|---|
| `results/metrics_history.csv` | Metriche per step, Fase 2 + Fase 3 con fact-checker |
| `results/metrics_history_control.csv` | Metriche per step, Fase 2 + Fase 3 di controllo |
| `results/phase3_report.json` | Report finale della run con fact-checker (seed CELF, reach, copertura) |
| `results/phase3_report_control.json` | Report finale della run di controllo |
| `results/pipeline_summary_control.json` | Riepilogo dell'intera pipeline per la run di controllo |
| `results/phase3_comparison.json` | Confronto intervento − controllo prodotto da `scripts/compare_phase3.py` |

### Conclusioni

1. Gli agenti LLM producono una dinamica di diffusione realistica: crescita rapida, saturazione intorno al 41% e comparsa spontanea di resistenti.
2. La co-evoluzione genera echo chamber di **opinione** (assortatività da 0.27 a 0.54) mentre la struttura per community si indebolisce: misurarle solo con l'ECI strutturale le avrebbe fatte sembrare in diminuzione.
3. Un intervento tardivo con 20 fact-checker (0,4% dei nodi) protegge una parte dei suscettibili ma lascia quasi invariata la popolazione infetta già consolidata. Con le regole del modello, il fact-checking funziona come prevenzione più che come cura.
4. L'intervento non riduce la polarizzazione: la rafforza leggermente, perché sposta nodi neutrali su posizioni nette.

> **Nota sui report.** Nei file `phase3_report*.json` i campi `delta_avg_clustering`, `delta_num_connected_components` e gli altri delta topologici misurano il cambiamento rispetto alla Fase 0, non rispetto allo step 99, a causa di un errore di cache corretto dopo la run (vedi `CHANGELOG_FIX.md`, quarta revisione). I valori finali, gli stati, ECI, modularità e le metriche di opinione sono corretti.

## Limitazioni e Sviluppi Futuri

- **Una sola realizzazione.** Ogni configurazione viene eseguita una volta: con il costo attuale (circa 6 minuti per step) non è stato possibile ripetere le run con seed diversi. L'effetto sui suscettibili (−136 nodi) è netto; quelli sugli infetti (−16) e sull'assortatività (+0.013) sono piccoli e andrebbero confermati con più repliche.
- **Frammentazione.** Il rewiring a densità costante ha staccato dalla rete 45 piccoli gruppi di nodi. Un vincolo di connettività sulle rimozioni renderebbe la rete più realistica.
- **Ripetibilità dell'LLM.** Il seed per richiesta rende ripetibili le risposte a parità di prompt, ma vLLM non garantisce il determinismo completo con batching variabile.
- **Obiettivo di selezione miope.** L'obiettivo `threshold` considera l'effetto immediato dei fact-checker sui vicini, non la dinamica futura (ricadute, rewiring, risposte dell'LLM).
- **Community statiche.** ECI e modularità usano le community della Fase 0; i cambiamenti di opinione sono misurati dalle metriche di assortatività e omofilia.
- **Scala del modello.** Llama 3 8B quantizzato AWQ è un compromesso dettato dalla GPU disponibile (T4).
- Sviluppi possibili: **GNN Explainer** per interpretare quali legami guidino la polarizzazione; attivazione più precoce dell'intervento; agenti LLM con "personalità" diverse.

## Riferimenti

- Dataset: [Open Graph Benchmark — `ogbl-collab`](https://ogb.stanford.edu/docs/linkprop/#ogbl-collab)
- Hamilton et al., *Inductive Representation Learning on Large Graphs* (GraphSAGE)
- Kempe, Kleinberg, Tardos, *Maximizing the Spread of Influence through a Social Network* (base teorica di Influence Maximization / CELF)
- Leskovec et al., *Cost-effective Outbreak Detection in Networks* (algoritmo CELF)
- Serving LLM: [vLLM](https://github.com/vllm-project/vllm)

---

*README generato a partire dalla relazione tecnica del progetto (`progettoASM_relazione.tex`). Per l'analisi completa — inclusi grafici, tabelle di confronto dettagliate e discussione critica dei risultati — fare riferimento alla relazione integrale.*
