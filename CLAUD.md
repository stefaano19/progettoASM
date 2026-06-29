# CLAUD.md — Bussola Tecnico-Scientifica
## Echo Chamber Co-Evolution Framework

> **Tipo:** Documento interno di riferimento tecnico
> **Aggiornamento:** Mantenere aggiornato ad ogni milestone completata
> **Leggi prima di ogni sessione di sviluppo**

---

## 1. Architettura Logica

### 1.1 Principio Fondante: Co-evoluzione Continua

Il sistema non e' una pipeline sequenziale. E' un **sistema dinamico a retroazione** in cui tre sottosistemi si influenzano mutuamente ad ogni step temporale:

```
AGENTI LLM  <-->  TOPOLOGIA (GNN)  <-->  STATI (Linear Threshold)
     ^                                          |
     |                   CELF                   |
     +------------------------------------------+
```

- **Non** si addestrano prima gli agenti e poi la GNN.
- **Non** si fa rewiring separato dalla diffusione.
- Il loop di co-evoluzione e' l'unita' atomica del sistema.

### 1.2 I Tre Livelli

#### Livello 1 — Cognitivo (Agenti LLM)

Ogni nodo del grafo e' un agente generativo. L'agente:
- Riceve in input: embedding corrente, stato del vicinato, centralita', storico post visibili.
- Produce in output: opinione testuale + stato aggiornato (S/I/R) + delta embedding.
- Il meccanismo di transizione di stato e' il **Linear Threshold Model**, ma la soglia e' dinamicamente modulata dall'output qualitativo dell'LLM.

**Stati possibili:**
| Stato | Etichetta | Descrizione |
|-------|-----------|-------------|
| S | Susceptible | Non ancora "infettato" dalla narrazione polarizzante |
| I | Infected | Crede e diffonde attivamente la narrazione |
| R | Resistant | Esposto ma critico; potenziale vettore di fact-checking |
| F | Fact-Checker | Agente seed iniettato da CELF per contrastare I |

#### Livello 2 — Strutturale (GNN + Link Prediction)

La GNN (GraphSAGE) lavora sugli embedding perturbati dagli agenti:
- **Input:** graph at time t + embedding aggiornati dagli agenti.
- **Output:** score di probabilita' per ogni potenziale arco (esistente o nuovo).
- **Soglia di rewiring:** gli archi con score < threshold_remove vengono eliminati; i candidati con score > threshold_add vengono aggiunti.
- La GNN viene ri-addestrata (fine-tuned) ad ogni N step sul grafo corrente.

#### Livello 3 — Intervento (CELF)

CELF (Cost-Effective Lazy Forward) per Influence Maximization:
- **Budget:** k nodi seed (fact-checker) da iniettare.
- **Funzione di spread:** numero atteso di nodi raggiunti dalla narrazione "verita'" in T passi.
- **Attivazione:** dopo un numero configurabile di step, o al superamento di una soglia di infection rate.

### 1.3 Ciclo Temporale Completo

```
Step t:
  per ogni nodo n in ordine randomizzato:
    1. feed  = get_neighbourhood_posts(n, window=W)
    2. ctx   = {embedding[n], state[n], centrality[n], feed}
    3. output = LLM.reason(ctx)            -> {opinion, new_state, delta_emb}
    4. embedding[n] += delta_emb           -> perturbazione
    5. state[n]     = LT_transition(n, output.new_state)  -> transizione

  GNN.forward(G_t, embeddings_t)           -> link_scores
  G_{t+1} = rewire(G_t, link_scores)       -> nuova topologia

  if step % celf_interval == 0:
    seeds = CELF.select(G_{t+1}, budget=k)
    inject_fact_checkers(seeds)

  log_checkpoint(step, G_{t+1}, states, embeddings, metrics)
```

---

## 2. Roadmap a 4 Fasi

### Fase 0 — Setup & Baseline

**Obiettivo:** ambiente funzionante, dati caricati, metriche baseline calcolate.

| Task | Modulo | Output |
|------|--------|--------|
| Download ogbl-collab | `src/graph/data_loader.py` | `data/raw/` |
| Estrazione sottografo (1-5%) | `src/graph/extractor.py` | `data/processed/subgraph.gpickle` |
| Metriche topologiche baseline | `src/graph/metrics.py` | densita', clustering, diameter |
| Community detection (Louvain) | `src/graph/community.py` | node -> community_id map |
| Config loader & seed manager | `src/utils/config.py` | `Config` dataclass |
| Logger JSONL | `src/utils/logger.py` | `results/logs/` |
| Test modulo graph | `tests/test_graph.py` | CI verde |

**Criteri di completamento Fase 0:**
- Il sottografo e' caricabile e le metriche sono riproducibili con seed fisso.
- Community detection produce community coerenti (Q > 0.3 sul sottografo).
- Tutti i test in `tests/test_graph.py` passano.

---

### Fase 1 — Logica Agente (Infezione)

**Obiettivo:** agenti LLM attivi, "pazienti zero" identificati, primo ciclo di diffusione.

| Task | Modulo | Output |
|------|--------|--------|
| LLM Client (Gemini/Ollama) | `src/agents/llm_client.py` | LLMClient class |
| Definizione stati + transizioni LT | `src/agents/state_machine.py` | StateTransition logic |
| Prompt engineering agente | `src/agents/prompts.py` | system/user prompt templates |
| Classe Agent | `src/agents/agent.py` | Agent class |
| Identificazione pazienti zero | `src/agents/seeder.py` | PageRank/Betweenness/Katz |
| Token budget tracker | `src/agents/llm_client.py` | TokenBudget class |
| Test agente singolo | `tests/test_agent.py` | mock LLM, validazione output JSON |

**Criteri di completamento Fase 1:**
- Un agente singolo esegue un ciclo completo (percezione → cognizione → transizione).
- L'output LLM e' sempre JSON valido (con retry/fallback).
- I pazienti zero sono selezionati in modo riproducibile (seed fisso).

---

### Fase 2 — Dinamiche di Rete (Co-evoluzione)

**Obiettivo:** loop temporale completo con co-evoluzione agenti↔GNN.

| Task | Modulo | Output |
|------|--------|--------|
| Embedding iniziali (node2vec o feat. ogb) | `src/gnn/embeddings.py` | embedding matrix |
| Modello GraphSAGE | `src/gnn/model.py` | GraphSAGE class (PyG) |
| Training loop GNN | `src/gnn/trainer.py` | trained model + link scores |
| Rewiring logic | `src/gnn/rewirer.py` | add/remove edges |
| Orchestratore principale | `src/orchestrator.py` | SimulationOrchestrator |
| Network Manager | `src/graph/network_manager.py` | NetworkManager class |
| Checkpoint & logging | `src/utils/checkpoint.py` | .gpickle + .jsonl per step |
| Test co-evoluzione (5 step) | `tests/test_coevolution.py` | stato del grafo coerente |

**Criteri di completamento Fase 2:**
- Il loop gira per N step configurabili senza errori.
- Le metriche (Q, ECI, BP) cambiano in modo osservabile nel tempo.
- I checkpoint sono ricaricabili e la simulazione e' riprendibile.

---

### Fase 3 — Deployment & Fact-Checking

**Obiettivo:** CELF funzionante, esportazione Kaggle, risultati finali.

| Task | Modulo | Output |
|------|--------|--------|
| Implementazione CELF | `src/influence/celf.py` | seed selection |
| Iniezione fact-checker | `src/influence/injector.py` | stato F propagato |
| Metriche disgregazione | `src/influence/metrics.py` | FCS, delta-Q, delta-ECI |
| Notebook Kaggle | `notebooks/kaggle_full_run.ipynb` | pronto per full-scale |
| Export risultati | `results/` | figure + JSONL finali |
| Report finale | `results/report.md` | analisi e interpretazione |

---

## 3. Schema File Definitivo

```
project-root/
|-- README.md
|-- CLAUD.md                        <- questo file
|-- config.yaml
|-- requirements.txt
|-- .gitignore
|
|-- data/
|   |-- raw/                        <- ogbl-collab originale (.gitignored)
|   +-- processed/
|       |-- subgraph.gpickle        <- sottografo estratto
|       |-- embeddings.npy          <- embedding matrice (n x d)
|       +-- community_map.json      <- {node_id: community_id}
|
|-- notebooks/
|   +-- kaggle_full_run.ipynb
|
|-- src/
|   |-- __init__.py
|   |-- orchestrator.py             <- loop principale co-evoluzione
|   |
|   |-- graph/
|   |   |-- __init__.py
|   |   |-- data_loader.py          <- download + caricamento ogbl-collab
|   |   |-- extractor.py            <- campionamento sottografo
|   |   |-- metrics.py              <- densita', clustering, diameter, Q, ECI
|   |   |-- community.py            <- Louvain community detection
|   |   +-- network_manager.py      <- CRUD su nodi/archi, feed vicinato
|   |
|   |-- agents/
|   |   |-- __init__.py
|   |   |-- agent.py                <- classe Agent (percezione, cognizione, azione)
|   |   |-- state_machine.py        <- stati S/I/R/F + Linear Threshold
|   |   |-- prompts.py              <- template system/user prompt
|   |   |-- llm_client.py           <- wrapper Gemini/OpenAI/Ollama + TokenBudget
|   |   +-- seeder.py               <- identificazione pazienti zero (PageRank, ecc.)
|   |
|   |-- gnn/
|   |   |-- __init__.py
|   |   |-- model.py                <- GraphSAGE (PyTorch Geometric)
|   |   |-- trainer.py              <- training/fine-tuning loop
|   |   |-- embeddings.py           <- inizializzazione embedding
|   |   +-- rewirer.py              <- soglie add/remove + aggiornamento grafo
|   |
|   |-- influence/
|   |   |-- __init__.py
|   |   |-- celf.py                 <- Cost-Effective Lazy Forward
|   |   |-- injector.py             <- iniezione agenti fact-checker
|   |   +-- metrics.py              <- FCS, delta-Q, delta-ECI post-intervento
|   |
|   +-- utils/
|       |-- __init__.py
|       |-- config.py               <- YAML loader -> Config dataclass
|       |-- logger.py               <- JSONL structured logger
|       |-- seed.py                 <- seed manager (numpy, torch, random)
|       +-- checkpoint.py           <- save/load .gpickle + embedding
|
|-- results/
|   |-- figures/
|   +-- logs/
|
+-- tests/
    |-- __init__.py
    |-- test_graph.py
    |-- test_agent.py
    |-- test_coevolution.py
    +-- test_influence.py
```

---

## 4. Standard di Sviluppo

### 4.1 Portabilita' Locale <-> Kaggle

- **Nessun percorso assoluto** hardcoded. Tutti i path derivano da `config.yaml` + `Path(__file__).parent.parent`.
- `config.yaml` ha due sezioni distinte: `[local]` e `[kaggle]`. Il loader seleziona in base a `execution.mode`.
- Il notebook Kaggle importa `src/` dopo aver clonato il repo: `import sys; sys.path.insert(0, '/kaggle/working/project-root')`.

### 4.2 Gestione Dipendenze

```
# requirements.txt: dipendenze base (CPU)
networkx>=3.2
torch>=2.1               # CPU wheels su Mac
torch-geometric>=2.4     # installazione separata (vedi README)
ogb>=1.3.6               # per ogbl-collab loader
python-louvain>=0.16
google-generativeai>=0.7
openai>=1.0
pyyaml>=6.0
numpy>=1.26
tqdm>=4.66
```

Su Kaggle il runtime CUDA e' pre-installato; si usa `requirements.txt` con `--extra-index-url` per PyG CUDA.

### 4.3 Riproducibilita'

Ogni esecuzione deve loggare:
```json
{
  "run_id": "uuid",
  "timestamp": "ISO8601",
  "config_hash": "sha256 di config.yaml",
  "random_seed": 42,
  "python_version": "3.13.3",
  "git_commit": "sha"
}
```

Il seed manager (`src/utils/seed.py`) setta:
```python
random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

### 4.4 Token Budget (LLM)

| Parametro | Valore default |
|-----------|----------------|
| `warn_at` | 50,000 tokens |
| `hard_limit` | 200,000 tokens |
| Conteggio | input + output, per step e cumulativo |
| Log | ogni step nel JSONL |

### 4.5 Ciclo di Test

```bash
# Prima di ogni commit o push su Kaggle:
pytest tests/ -v --tb=short

# Test singola fase:
pytest tests/test_graph.py -v          # Fase 0
pytest tests/test_agent.py -v          # Fase 1
pytest tests/test_coevolution.py -v    # Fase 2
pytest tests/test_influence.py -v      # Fase 3
```

Ogni test usa il sottografo ridotto (N <= 100 nodi) e un mock LLM per azzerare costi API.

---

## 5. Stato di Progetto

> **Aggiornare questa sezione ad ogni milestone completata**

| Fase | Titolo | Stato | Data completamento |
|------|--------|--------|-------------------|
| **0** | Setup & Baseline | ✅ Completata | 2026-06-28 |
| **1** | Logica Agente (Infezione) | ✅ Completata | 2026-06-28 |
| **2** | Dinamiche di Rete (Co-evoluzione) | ✅ Completata | 2026-06-29 |
| **3** | Deployment & Fact-Checking | ✅ Completata | 2026-06-29 |

**Fase corrente:** Tutte le fasi completate — pipeline pronta
**Ultimo aggiornamento:** 2026-06-29
**Entry point per fase:**
  - `python phase0_run.py --skip-download`
  - `python phase1_run.py`
  - `python phase2_run.py --steps 10`
  - `python phase3_run.py --budget-k 10`
  - Notebook completo: `notebooks/kaggle_full_run.ipynb`
**Test Fase 0:** 31/31 PASSED
**Test Fase 1:** 44/44 PASSED (`pytest tests/test_agent.py -v`)
**Test totali:** 75/75 PASSED (`pytest tests/ -v`)

---

## 6. Decisioni Architetturali Chiave (Decision Log)

| Data | Decisione | Motivazione |
|------|-----------|-------------|
| 2026-06-28 | Dataset: `ogbl-collab` invece di Cora/Citeseer | Scala reale (>2M archi), ground truth temporale, compatibile OGB |
| 2026-06-28 | GNN: GraphSAGE invece di GAT | Inductive learning (nodi nuovi durante rewiring), scalabilita' mini-batch |
| 2026-06-28 | LT Model guidato da LLM invece di soglia fissa | Variabilita' comportamentale realistica, non riducibile a parametro scalare |
| 2026-06-28 | CELF invece di greedy IM | Complessita' O(k * n * R) vs O(n^2 * R); fondamentale per grafi grandi |
| 2026-06-28 | Struttura `src/` modulare con `config.yaml` | Portabilita' locale<->Kaggle senza modifiche al codice |
