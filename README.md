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
| Influence Maximization | **CELF** (Cost-Effective Lazy Forward) | Selezione submodulare dei nodi seed per il fact-checking |
| Dataset | `ogbl-collab` (Open Graph Benchmark) | Rete di co-autoraggio accademico, ~235k nodi / ~1.2M archi |

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
│   │   ├── metrics.py             # Centralità, modularità, ECI, belief polarisation
│   │   └── network_manager.py     # Layer unico di astrazione/persistenza del grafo dinamico
│   ├── agents/
│   │   ├── agent.py               # Agente cognitivo (percezione → cognizione → azione)
│   │   ├── llm_client.py          # Wrapper LLM portabile (vLLM/Ollama/Gemini + cache su disco)
│   │   ├── seeder.py              # Selezione dei "pazienti zero"
│   │   └── state_machine.py       # Transizioni S/I/R/F (Linear Threshold modulato)
│   ├── gnn/
│   │   ├── embeddings.py          # EmbeddingManager (Word2Vec)
│   │   ├── model.py               # GraphSAGEModel (PyTorch Geometric / NumPy fallback)
│   │   ├── trainer.py             # GNNTrainer — training + link prediction
│   │   └── rewirer.py             # Applica soglie/filtri di omofilia per il rewiring
│   ├── influence/
│   │   ├── celf.py                # Algoritmo CELF (Influence Maximization)
│   │   ├── injector.py            # FactCheckerInjector
│   │   └── metrics.py             # Fact-Checker Spread, Intervention Delta
│   └── utils/
│       ├── logger.py              # SimLogger (log JSONL)
│       ├── checkpoint.py          # CheckpointManager (resume cross-sessione)
│       └── seed.py                # Riproducibilità (set_all_seeds)
└── results/
    ├── metrics_history.csv        # Storico metriche per step
    └── phase3_report.json         # Report finale dell'intervento CELF
```

> La struttura sopra riflette i moduli descritti nella relazione tecnica del progetto; verificane i percorsi esatti nel repository, che potrebbero differire leggermente.

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
| `PHASE2_STEPS` | Numero di **nuovi** step da eseguire in Fase 2 in questa sessione (non il totale cumulato) | `0` (se si riprende da checkpoint) |
| `PHASE3_STEPS` | Step di simulazione post-intervento in Fase 3 | `30` |
| `CELF_BUDGET_K` | Numero di fact-checker da iniettare | `20` |
| `SAMPLING_STRATEGY` | Strategia di campionamento del sottografo | `forest_fire` \| `bfs_seed` \| `random_walk` \| `random_nodes` |
| `FOREST_FIRE_PROB` | Forward probability del Forest Fire Sampling (0.4–0.7) | `0.5` |
| `TARGET_NODES` | Dimensione del sottografo campionato | `5000` |
| `RESUME_FROM_CKPT` | Riprendi da un checkpoint salvato (cross-sessione, utile su Kaggle) | `True` |

## Esecuzione della Pipeline

L'intera pipeline è orchestrata dal notebook `kaggle_full_run.ipynb`, suddiviso in 4 fasi eseguite in sequenza nella stessa sessione (o riprese via checkpoint):

```
Fase 0 → Fase 1 → Fase 2 → Fase 3
```

Ogni fase salva automaticamente checkpoint (`.pkl`) e metriche (`metrics_history.csv`), così l'esecuzione può essere interrotta e ripresa — utile per superare i limiti di tempo delle sessioni Kaggle gratuite.

## Le Quattro Fasi

### Fase 0 — Setup, Data Ingestion e Baseline
Inizializza il server vLLM, scarica/carica `ogbl-collab`, estrae un sottografo di 5.000 nodi tramite **Forest Fire Sampling**, rileva le community (Label Propagation) e calcola le metriche baseline (centralità, modularità, Echo Chamber Index).

- **Validazione strutturale:** il sottografo preserva la distribuzione dei gradi a legge di potenza e le proprietà strutturali chiave del grafo originale (232.865 nodi → 5.000 nodi, rapporto di campionamento ~2.1%).
- **Baseline:** 415 community, Modularity Q = 0.8265, Echo Chamber Index = 0.8318.

### Fase 1 — Livello Cognitivo e Agenti LLM
Instanzia gli agenti (`agent.py`), il client LLM (`llm_client.py`), seleziona i "pazienti zero" (`seeder.py`, 15% della rete) e attiva la macchina a stati (`state_machine.py`) che governa le transizioni S → I → R → F secondo un modello Linear Threshold modulato dalla suscettibilità cognitiva valutata dall'LLM.

### Fase 2 — Dinamiche di Rete e Co-evoluzione
Il `SimulationOrchestrator` esegue il ciclo ricorsivo: **ciclo Agenti** (chiamate LLM asincrone e batched) → **ciclo GNN** (training + link prediction) → **ciclo di Rewiring** (aggiunta/rimozione archi in base all'omofilia ideologica). Checkpoint frequenti garantiscono resilienza cross-sessione.

### Fase 3 — Intervento e Fact-Checking (CELF)
Seleziona tramite l'algoritmo **CELF** i nodi ottimali per massimizzare la diffusione del messaggio correttivo, li converte in Fact-Checker (stato F) e fa avanzare la simulazione per $N$ step post-intervento, confrontando le metriche prima/dopo.

## Risultati Principali

### Baseline (Fase 0, step 0)

| Metrica | Valore |
|---|---|
| Nodi / Archi | 5.000 / 26.246 |
| Densità | 0.0021 |
| Grado medio | 10.50 |
| Clustering medio | 0.6988 |
| Modularity Q | 0.8265 |
| Echo Chamber Index | 0.8318 |

### Prima vs. Dopo l'intervento CELF (30 step post-intervento, 20 fact-checker)

| Metrica | Prima | Dopo | Delta |
|---|---|---|---|
| Infection Rate | 0.5146 | 0.5690 | +0.0544 |
| Echo Chamber Index | 0.8114 | 0.8092 | −0.0022 |
| Modularity Q | 0.7925 | 0.7898 | −0.0027 |
| Belief Polarisation | 0.5577 | 0.9758 | +0.4181 |
| Nodi F (Fact-Checker) | 0 | 20 | +20 |

**Sintesi:** l'intervento CELF utilizza correttamente l'intero budget richiesto, ma con soli 20 fact-checker su 5.000 nodi (0.4% della popolazione) non riesce a invertire il trend epidemico — anzi, allungare la finestra post-intervento avvantaggia anche la naturale prosecuzione del contagio. L'effetto più marcato si osserva a livello **cognitivo** (Belief Polarisation quasi raddoppiata) più che strutturale (ECI e Modularity restano quasi invariati). L'LLM reale (vLLM) è attivo in tutte le fasi, incluso il post-intervento: il risultato non è quindi un artefatto di semplificazione del ragionamento, ma un effetto genuino della sproporzione tra budget e scala dell'infezione.

## Limitazioni e Sviluppi Futuri

- Il budget CELF testato (fino a 20 nodi) resta uno o più ordini di grandezza sotto la soglia necessaria per un contenimento strutturale misurabile su una rete di 5.000 nodi con Infection Rate già > 50%.
- Un'attivazione più precoce dell'intervento (prima che l'infezione superi ampiamente metà della popolazione) potrebbe risultare più efficace di un budget maggiore a parità di tempistica.
- Integrazione di **GNN Explainer** per interpretare visivamente quali legami guidino effettivamente la polarizzazione.
- Test con architetture di agenti LLM diverse, per valutare l'effetto di differenti "personalità" algoritmiche sulla velocità di convergenza verso l'omofilia.

## Riferimenti

- Dataset: [Open Graph Benchmark — `ogbl-collab`](https://ogb.stanford.edu/docs/linkprop/#ogbl-collab)
- Hamilton et al., *Inductive Representation Learning on Large Graphs* (GraphSAGE)
- Kempe, Kleinberg, Tardos, *Maximizing the Spread of Influence through a Social Network* (base teorica di Influence Maximization / CELF)
- Leskovec et al., *Cost-effective Outbreak Detection in Networks* (algoritmo CELF)
- Serving LLM: [vLLM](https://github.com/vllm-project/vllm)

---

*README generato a partire dalla relazione tecnica del progetto (`progettoASM_relazione.tex`). Per l'analisi completa — inclusi grafici, tabelle di confronto dettagliate e discussione critica dei risultati — fare riferimento alla relazione integrale.*