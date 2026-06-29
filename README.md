# Echo Chamber Co-Evolution Framework
### GNN-Driven Link Prediction × LLM Generative Agents × Influence Maximization

> **Corso:** Analisi di Social Network e Media
> **Dataset:** `ogbl-collab` — Open Graph Benchmark
> **Ambiente:** Sviluppo locale (Mac/CPU) → Deployment cloud (Kaggle/GPU)

---

## Panoramica

Questo progetto sviluppa un **framework di co-evoluzione dinamica** per modellare la formazione, l'evoluzione e il contrasto delle *echo chamber* all'interno di una rete sociale accademica.

L'innovazione è **triplice e co-evolutiva**: tre livelli del sistema si influenzano in un ciclo continuo, senza mai essere separati in pipeline indipendenti.

| Livello | Componente | Ruolo |
|---------|------------|-------|
| **Cognitivo** | Agenti generativi LLM | Producono opinioni testuali, decidono stati di "infezione", perturbano gli embedding |
| **Strutturale** | GNN (GraphSAGE) + Link Prediction | Ricalcola probabilità di link sugli embedding perturbati → rewiring topologico |
| **Intervento** | Influence Maximization (CELF) | Inietta agenti fact-checker nei nodi a massimo impatto per disgregare le echo chamber |

---

## Architettura del Sistema

```
+------------------------------------------------------------------+
|                    LOOP DI CO-EVOLUZIONE                         |
|                                                                  |
|  +----------+    opinioni/     +-------------+                   |
|  |  Agenti  | --embedding--->  |     GNN     |                   |
|  |   LLM    |                  |  GraphSAGE  |                   |
|  +----+-----+                  +------+------+                   |
|       |   nuova topologia            |  nuovi link              |
|       |<------------------------------+                          |
|       |                                                          |
|       v                                                          |
|  +----------------------+                                        |
|  |  Linear Threshold    | <- stato vicinato + output LLM        |
|  |  (transizioni)       |                                        |
|  +----------------------+                                        |
|                |                                                 |
|                v                                                 |
|  +----------------------+                                        |
|  |  CELF / Influence    | -> inietta fact-checker nodes         |
|  |  Maximization        |                                        |
|  +----------------------+                                        |
+------------------------------------------------------------------+
```

### Ciclo Temporale (un "tick")

1. **Percezione** — Ogni agente LLM legge: stato vicinato, embedding corrente, centralita' del nodo.
2. **Cognizione** — L'LLM produce un'opinione testuale e aggiorna il proprio stato *(suscettibile / infetto / resistente)*.
3. **Perturbazione** — L'output dell'agente modifica il feature vector del nodo nel grafo.
4. **Rewiring** — La GNN ricalcola le probabilita' di link sugli embedding perturbati; archi vengono aggiunti/rimossi.
5. **Transizione** — Il modello Linear Threshold (guidato dall'output LLM) calcola le nuove transizioni di stato.
6. **Checkpoint** — Log di stato (agenti + topologia), salvataggio metriche (Q-score, ECI, Polarisation Index).
7. *(opzionale)* **CELF** — Al raggiungimento di una soglia, viene attivato il modulo di Influence Maximization.

---

## Struttura del Progetto

```
project-root/
|-- README.md                  <- questo file
|-- CLAUD.md                   <- bussola tecnica (architettura, roadmap, standard)
|-- config.yaml                <- unica fonte di verita' per tutti i parametri
|
|-- data/
|   |-- raw/                   <- dataset scaricato (ogbl-collab, .gitignored)
|   +-- processed/             <- sottografi, embedding, artefatti pre-processati
|
|-- notebooks/
|   +-- kaggle_full_run.ipynb  <- notebook Kaggle (full-scale, GPU)
|
|-- src/
|   |-- graph/                 <- estrazione sottografo, metriche, community detection
|   |-- agents/                <- logica agente LLM, stati, prompt engineering
|   |-- gnn/                   <- modello GraphSAGE, training, rewiring
|   |-- influence/             <- CELF, Influence Maximization
|   +-- utils/                 <- config loader, logging, seed management
|
|-- results/
|   |-- figures/               <- grafici generati (polarisation curve, modularity)
|   +-- logs/                  <- JSONL: log decisioni agenti + metriche per step
|
+-- tests/                     <- test minimi per ogni modulo src/
```

---

## Setup Ambiente

### Prerequisiti

- Python >= 3.10
- Homebrew (Mac) con `pip`

### Installazione Locale (Mac/CPU)

```bash
# Clona il repository
git clone <repo-url>
cd project-root

# Attiva virtual environment esistente (o creane uno nuovo)
source .venv/bin/activate

# Installa dipendenze
pip install -r requirements.txt
```

### Variabili d'Ambiente

```bash
# Richiesto per backend LLM cloud (se llm.backend = "api")
export GEMINI_API_KEY="your-key-here"
# oppure
export OPENAI_API_KEY="your-key-here"
```

### Configurazione

Modifica `config.yaml` per switchare tra modalita':

```yaml
execution:
  mode: "local"          # "local" | "kaggle"

llm:
  backend: "local"       # "api" | "local" (Ollama)

simulation:
  subgraph_size: 500     # 500 in locale, full su Kaggle
```

---

## Workflow Locale vs Kaggle

```
Locale (Mac/CPU)                      Kaggle (GPU)
-----------------                     ------------
  Fase 0: Setup           git push ->  kaggle_full_run.ipynb
  Fase 1: Agenti          (GitHub)       !pip install -r requirements.txt
  Fase 2: GNN Proto                      clone repo, set config kaggle
  Fase 3: CELF proto                     run full ogbl-collab
                                         export results/ -> download
  <- pull results <-----------------   analisi locale + grafici
```

**Regola d'oro:** il codice in `src/` e' identico in entrambi gli ambienti. Solo `config.yaml` cambia.

---

## Metriche di Valutazione

| Metrica | Sigla | Descrizione |
|---------|-------|-------------|
| Modularity | **Q** | Coesione delle community; Q alto indica echo chamber piu' forti |
| Echo Chamber Index | **ECI** | Frazione di archi intra-comunita' per nodo |
| Belief Polarisation | **BP** | Varianza normalizzata degli stati di "infezione" |
| Infection Rate | **IR** | Percentuale nodi nello stato infetto al passo t |
| Fact-check Spread | **FCS** | Percentuale nodi raggiunti dal seed CELF dopo k passi |

---

## Roadmap Rapida

| Fase | Titolo | Status |
|------|--------|--------|
| **0** | Setup & Baseline | Non iniziata |
| **1** | Logica Agente (Infezione) | Non iniziata |
| **2** | Dinamiche di Rete (Co-evoluzione) | Non iniziata |
| **3** | Deployment & Fact-Checking | Non iniziata |

> Dettaglio completo in `CLAUD.md`

---

## Licenza & Citazione

Progetto accademico — Analisi di Social Network e Media.
Dataset: Hu et al., 2020 — Open Graph Benchmark (https://arxiv.org/abs/2005.00687)
