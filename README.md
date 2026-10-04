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
- [Conclusioni](#conclusioni)
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
| Dataset | `ogbl-collab` (Open Graph Benchmark) | Rete di co-autoraggio accademico, 235.868 nodi / 967.632 archi (non orientati, deduplicati) |

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
│   │   ├── prompts.py             # Prompt di sistema/utente degli agenti
│   │   ├── seeder.py              # Selezione dei "pazienti zero"
│   │   └── state_machine.py       # Transizioni S/I/R/F (Linear Threshold modulato)
│   ├── gnn/
│   │   ├── embeddings.py          # EmbeddingManager (Word2Vec)
│   │   ├── model.py               # GraphSAGEModel (PyTorch Geometric / NumPy fallback)
│   │   ├── trainer.py             # GNNTrainer — training + link prediction
│   │   └── rewirer.py             # Applica le soglie di score GNN (con vincoli di sicurezza)
│   ├── influence/
│   │   ├── celf.py                # Algoritmo CELF (Influence Maximization)
│   │   ├── injector.py            # FactCheckerInjector
│   │   └── metrics.py             # Fact-Checker Spread, Intervention Delta
│   └── utils/
│       ├── logger.py              # SimLogger (log JSONL)
│       ├── checkpoint.py          # CheckpointManager (resume cross-sessione)
│       ├── config.py              # Caricamento config.yaml in dataclass
│       └── seed.py                # Riproducibilità (set_all_seeds)
├── phase0_run.py … phase3_run.py   # Entry point delle singole fasi
├── scripts/check_propagation.py
├── tests/                     # Test pytest (incl. test_fixes.py)
└── results/                   # Generati a runtime: metrics_history.csv,
                               # phase2_report.json, phase3_report.json, checkpoints/
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

I valori seguenti provengono dall'ultima esecuzione di `notebooks/kaggle_full_run.ipynb` (LLM reale `llama-3-8b-instruct-awq` su vLLM, seed 42), successiva alle correzioni di `CHANGELOG_FIX.md`. La Fase 2 è stata eseguita in due sessioni Kaggle (step 0–57, poi resume da `ckpt_step_0057.pkl` per gli step 58–95). Parametri effettivi (override del notebook): 15% di pazienti zero (750 nodi), 15% di nodi attivati per step, rewiring ogni 2 step.

### Baseline (Fase 0, step 0)

| Metrica | Grafo originale (LCC) | Sottografo (Forest Fire) |
|---|---|---|
| Nodi / Archi | 232.865 / 961.883 | 5.000 / 26.246 |
| Grado medio | 8.3 | 10.50 |
| Clustering medio | 0.7204 | 0.6988 |
| Modularity Q (Label Propagation) | 0.6957 | 0.8265 |
| Community | 23.115 | 415 |
| Echo Chamber Index | — | 0.8318 |

Il campionamento preserva bene il clustering (−3%) ma produce un sottografo più modulare (+19% di Q) e con meno hub (grado max 119 contro 382).

### Co-evoluzione (Fase 2, step 0 → 95)

| Metrica | Step 0 | Step 58 | Step 95 |
|---|---|---|---|
| S / I / R | 4.250 / 750 / 0 | 1.043 / 1.958 / 1.999 | 956 / 1.974 / 2.070 |
| Infection Rate | 0.150 | 0.392 | 0.395 |
| Archi | 26.246 | 27.295 | 27.332 |
| Echo Chamber Index | 0.8318 | 0.8142 | 0.8131 |
| Modularity Q | 0.8265 | — | 0.7952 |
| Belief Polarisation | — | — | 0.2420 |
| Loss GNN | — | 0.542 | 0.537 |

- **Contagio:** l'infezione cresce da 15% a circa 39% nella prima metà della simulazione e poi entra in un **plateau**: tra lo step 58 e il 95 il numero di infetti varia di appena +16 nodi, mentre ogni step 3–5 suscettibili passano a R. A fine Fase 2 i resistenti (41%) superano gli infetti (39%).
- **Topologia:** il rewiring guidato dalla GNN è debole (+1.086 archi in 96 step, +4%; nella seconda sessione solo +37) e quasi solo additivo. Gli archi aggiunti sono in prevalenza inter-community, tanto che ECI e Q **diminuiscono** leggermente invece di aumentare.

### Fase 3 — intervento CELF (20 fact-checker)

CELF ha selezionato 20 nodi S (spread stimato in Independent Cascade: 65.9 nodi, 1.3% della rete), iniettati allo step 96 con Infection Rate pari a 0.3948. **Nell'ultima esecuzione `PHASE3_STEPS = 0`**: nessuno step post-intervento è stato simulato, quindi il confronto prima/dopo del notebook mostra delta nulli (l'unica differenza è S → F per i 20 seed). L'effetto dell'intervento sulla dinamica corretta va ancora misurato.

> **Run precedente (prima delle correzioni, non confrontabile):** con 30 step post-intervento l'Infection Rate saliva da 0.5146 a 0.5690. In quella versione però i fact-checker potevano agire solo sugli infetti con ≥25% di vicini F, per cui l'intervento non poteva avere effetto per costruzione, e la Belief Polarisation confrontava due codifiche diverse (+0.418 era un artefatto).

## Conclusioni

1. **La dinamica cognitiva domina su quella strutturale.** Gli agenti LLM determinano quasi tutta l'evoluzione del sistema (transizioni S → I/R), mentre la topologia cambia di pochi punti percentuali. Il "loop chiuso" di co-evoluzione è presente nel codice, ma nei parametri usati il feedback della rete sulla cognizione è debole.
2. **L'infezione si auto-limita senza intervento.** Partendo dal 15% di pazienti zero, la narrazione raggiunge circa il 40% della rete e poi si stabilizza. Il meccanismo di resistenza attiva (S → R quando l'LLM valuta bassa suscettibilità) fa sì che la maggioranza degli esposti diventi resistente anziché infetta.
3. **Nessuna formazione di echo chamber strutturali.** Contrariamente all'ipotesi di partenza, il rewiring non rafforza l'omofilia: ECI (0.832 → 0.813) e modularità (0.827 → 0.795) calano. Una parte di questo risultato dipende però da limiti del modello (vedi sotto): ECI è calcolato rispetto alle community *strutturali* fisse della Fase 0, non rispetto alle opinioni, e la generazione dei candidati per il rewiring è distorta.
4. **L'efficacia del fact-checking resta una domanda aperta.** Con 20 seed (0.4% della rete) lo spread atteso stimato da CELF è dell'1.3% dei nodi: un ordine di grandezza sotto la quota già infetta (39%). È ragionevole attendersi un effetto locale e limitato; per una conclusione quantitativa serve eseguire la Fase 3 con `PHASE3_STEPS > 0` e confrontarla con una run di controllo senza iniezione (stesso seed e stessi step).

## Limitazioni e Sviluppi Futuri

**Limiti noti dell'implementazione** (influenzano l'interpretazione dei risultati):

- `src/gnn/trainer.py` → `_generate_candidates` usa `random.Random(self._seed)` con seed fisso: ad ogni step vengono campionati **gli stessi 50 nodi sorgente** per i nuovi archi, quindi il rewiring esplora sempre la stessa piccola porzione di rete (nei log ricorrono gli stessi nodi, ad es. il 4983).
- `src/agents/state_machine.py` → le soglie LT per nodo sono estratte da un unico RNG nell'ordine di primo accesso e **non sono salvate nel checkpoint**: dopo ogni resume cross-sessione ogni nodo riceve una soglia diversa.
- `src/influence/metrics.py` → il *Fact-Checker Spread* è la raggiungibilità BFS: su un grafo connesso vale sempre 1.0 (avg reach = 5000) e non misura l'efficacia dell'intervento.
- **Echo Chamber Index** misura la segregazione rispetto alle community della Fase 0 e non dipende dagli stati degli agenti; per le echo chamber di opinione servirebbe una misura come l'assortatività degli stati (frazione di vicini I tra gli I).
- **Codifica del belief incoerente:** nella Belief Polarisation R vale 0.5 (vicino a I), mentre negli embedding R spinge nella direzione opposta a I.
- **CELF** ottimizza lo spread in un modello Independent Cascade diverso dal LT+LLM usato in simulazione; con 30 round Monte Carlo alcuni guadagni marginali risultano negativi (rumore di stima).
- L'iniezione è manuale: `FactCheckerInjector.should_activate` (`activation_threshold` = 0.4, `celf_interval`) non viene usato dal notebook, e con IR = 0.3948 la soglia non sarebbe stata raggiunta.

**Sviluppi futuri:**

- Eseguire la Fase 3 con 30+ step post-intervento e una run di controllo senza fact-checker, variando budget (20, 100, 250) e momento dell'iniezione (prima del plateau).
- Correggere i limiti sopra elencati e introdurre una metrica di echo chamber basata sulle opinioni.
- Integrare **GNN Explainer** per interpretare quali legami guidano il rewiring.
- Testare LLM e "personalità" degli agenti diverse per valutarne l'effetto sulla velocità di convergenza.

## Riferimenti

- Dataset: [Open Graph Benchmark — `ogbl-collab`](https://ogb.stanford.edu/docs/linkprop/#ogbl-collab)
- Hamilton et al., *Inductive Representation Learning on Large Graphs* (GraphSAGE)
- Kempe, Kleinberg, Tardos, *Maximizing the Spread of Influence through a Social Network* (base teorica di Influence Maximization / CELF)
- Leskovec et al., *Cost-effective Outbreak Detection in Networks* (algoritmo CELF)
- Serving LLM: [vLLM](https://github.com/vllm-project/vllm)

---

*README generato a partire dalla relazione tecnica del progetto (`progettoASM_relazione.tex`). Per l'analisi completa — inclusi grafici, tabelle di confronto dettagliate e discussione critica dei risultati — fare riferimento alla relazione integrale.*