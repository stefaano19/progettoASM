# Correzioni — revisione del codice

Tutte le correzioni sono coperte da `tests/test_fixes.py` (83 test totali, tutti verdi).
Verificato anche un run end-to-end con LLM mock: Fase 2 (step 0–2) → resume (step 3–4) → Fase 3 (step 5–6), senza step saltati o sovrascritti.

## Errori che alteravano i risultati

**1. Resume: agenti disallineati dal grafo** — `src/orchestrator.py`
Gli `Agent` venivano creati dal grafo appena ri-seminato e solo dopo il `NetworkManager` veniva sostituito con quello del checkpoint; lo stato interno degli agenti restava quello iniziale (in un test: 85 agenti su 100 sbagliati). Ora gli agenti sono creati dopo il resume, leggendo lo stato ripristinato. Ripristinate anche community map e metriche cumulative dal checkpoint. Aggiunto `orch.sync_agent_states(node_ids)` da usare dopo l'iniezione dei fact-checker (sostituisce l'assegnazione manuale di `agent._state`).

**2. Belief Polarisation con due codifiche diverse** — `src/graph/network_manager.py`, `src/graph/metrics.py`, `src/influence/metrics.py`
Il valore pre-intervento usava S=0, I=1, R=0.5, F=−0.5; quello post-intervento I=1, F=0.5, R=S=0. Ora esiste un'unica codifica (`STATE_TO_BELIEF`) usata ovunque. La normalizzazione usa la varianza massima della scala reale (−0.5…1) invece di 0.25 fisso, che saturava l'indice.

**3. `phase2_run.py --resume` ripartiva dallo step 0** — `phase2_run.py`, `src/utils/checkpoint.py`
Ora riparte da `orch.next_step`. I checkpoint sono ordinati e potati per numero di step (prima in ordine alfabetico: i nuovi checkpoint venivano cancellati subito).

**4. Direzione degli embedding casuale per ogni agente** — `src/agents/agent.py`
Ogni agente spingeva il proprio embedding in una direzione diversa, quindi il rewiring basato sulla similarità non rifletteva le opinioni. Ora la direzione è condivisa (seed globale) con una piccola componente individuale (10%).

**5. I fact-checker non potevano avere effetto** — `src/agents/state_machine.py`, `src/utils/config.py`, `config.yaml`
F agiva solo sugli I con ≥25% di vicini F. Aggiunta l'inoculazione: S → R se la frazione di vicini F ≥ `fc_protection_threshold` (default 0.10), è ≥ della frazione di vicini I e l'LLM non propone I. Entrambe le soglie sono ora in `config.yaml` (`influence.fc_protection_threshold`, `influence.fc_resistance_threshold`). In Fase 2 non ci sono nodi F, quindi la dinamica pre-intervento non cambia.
*Nota: è una scelta di modellazione; va descritta nella relazione.*

## Correzioni minori

- `phase3_run.py`: aggiunto `--no-mock-llm` (prima l'LLM era sempre mock); la Fase 3 parte da `orch.next_step` (prima saltava uno step).
- Notebook: stessa correzione dello step di partenza, uso di `sync_agent_states`, commento errato su `activation_probability` (0.15 = 15%).
- `src/influence/celf.py`: RNG con seed per candidato → selezione dei seed riproducibile.
- `src/gnn/rewirer.py`: il controllo "non isolare nodi" tiene conto delle rimozioni già scelte nello stesso step.
- `src/utils/config.py`: `min_resistance_exposure` e `resistance_susceptibility_cutoff` ora leggibili da `config.yaml` (prima venivano ignorati).
- Docstring della `StateMachine` allineata alla formula reale; README aggiornato (albero dei file, descrizione del rewirer, avviso sui risultati da rigenerare).

## Come eseguire

```bash
pip install -r requirements.txt
pytest -q                                   # 83 test

# Locale, LLM mock
python phase2_run.py --steps 10
python phase2_run.py --steps 10 --resume    # riprende dallo step 10
python phase3_run.py --steps 5 --budget-k 20

# LLM reale
python phase2_run.py --no-mock-llm ...
python phase3_run.py --no-mock-llm ...
```

Su Kaggle: rieseguire `notebooks/kaggle_full_run.ipynb`. I checkpoint vecchi restano compatibili: al resume gli agenti vengono ora ricostruiti dallo stato salvato.

---

# Seconda revisione

Dopo questa revisione **la Fase 2 va rieseguita da zero**: le correzioni 1, 2 e 5 cambiano la dinamica della simulazione. I checkpoint e i risultati delle run precedenti non sono più validi. Test: 94 (tutti verdi). Verificato end-to-end con LLM mock: Fase 2 → resume → Fase 3 con fact-checker e Fase 3 di controllo dallo stesso checkpoint → `scripts/compare_phase3.py`; eseguite anche le celle di Fase 3 del notebook in entrambe le modalità.

## Dinamica della simulazione

**1. Candidati del rewiring sempre uguali** — `src/gnn/trainer.py`
`_generate_candidates` ricreava `random.Random(self._seed)` a ogni chiamata: ogni step esplorava gli stessi 50 nodi sorgente e si fermava a 200 candidati, favorendo i primi. Ora il seed dipende dallo step, le sorgenti sono 500 (`gnn.rewire_candidate_sources`) e ogni sorgente propone al massimo 5 coppie (`gnn.rewire_candidates_per_source`).

**2. Il rewiring non rimuoveva mai archi** — `src/gnn/rewirer.py`
Il link predictor è addestrato sugli archi esistenti e assegna loro score alti, quindi nessuno scendeva sotto la soglia di rimozione: la rete si densificava e basta. Nuova modalità `gnn.rewire_mode: swap` (default): ogni arco aggiunto sostituisce l'arco esistente con lo score più basso, a densità costante. `max_new_edges_per_step` passa da 200 a 50 (circa 4.800 sostituzioni su 96 step, ≈ 18% degli archi). La vecchia logica resta disponibile con `rewire_mode: threshold`.

**3. Soglie dei nodi che cambiavano a ogni resume** — `src/agents/state_machine.py`
Le soglie erano estratte da un unico generatore nell'ordine in cui i nodi venivano interrogati e non erano salvate nel checkpoint. Ora la soglia dipende solo da (seed, id del nodo): identica in ogni sessione.

**4. Seed per richiesta all'LLM** — `src/agents/llm_client.py`
Ogni richiesta a vLLM usa un seed derivato dal prompt: a parità di prompt la risposta è la stessa, quindi la run con fact-checker e quella di controllo differiscono solo dove l'intervento cambia il contesto degli agenti.

**5. Selezione dei seed allineata alla simulazione** — `src/influence/celf.py`
CELF ottimizzava una cascata Independent Cascade, diversa dalle regole della simulazione, e con 30 round Monte Carlo produceva guadagni marginali negativi (−2.97, −2.07). Nuovo obiettivo `influence.celf_objective: threshold` (default): numero di nodi su cui la pressione dei fact-checker supera le soglie della StateMachine. È deterministico; poiché una funzione a soglia non è submodulare si usa il greedy completo (costo trascurabile). L'obiettivo `ic` resta disponibile, ora con common random numbers e guadagni troncati a 0.

## Metriche

**6. Reach dei fact-checker sempre 1.0** — `src/influence/metrics.py`
Contava i nodi raggiungibili senza limite di distanza: su una rete connessa valeva sempre 5000/5000. Ora conta i nodi entro `influence.reach_hops` salti (default 1). Nuova metrica `fc_effective_coverage`: frazione di nodi su cui la pressione F supera le soglie di transizione.

**7. Echo chamber sulle opinioni** — `src/graph/metrics.py`
L'ECI misura solo quanto gli archi restano nelle community della Fase 0 e non può rilevare echo chamber ideologiche. Nuove metriche `opinion_homophily` e `belief_assortativity`, registrate anche nel CSV e nel log di ogni step.

**8. Codifica delle opinioni** — `src/graph/network_manager.py`
R valeva +0.5 ("a metà strada verso I") mentre negli embedding R è spinto nella direzione opposta a I. Nuova codifica: S = 0, I = +1, R = F = −1.

## Notebook e script

- Nuovo parametro `CONTROL_RUN` per la Fase 3 di controllo (report `phase3_report_control.json`, etichetta `3_control` nel CSV). Nello script: `phase3_run.py --no-celf`.
- Nuovo `scripts/compare_phase3.py`: effetto dell'intervento = run con fact-checker − controllo.
- Soglia di attivazione `influence.activation_threshold` ora controllata nel notebook; `FORCE_INJECTION` permette di iniettare comunque e lo registra nel report.
- Con `PHASE3_STEPS = 0` (sessioni intermedie di Fase 2) il notebook non esegue più CELF e iniezione.
- Riepilogo: step totali di Fase 2 (non solo dell'ultima sessione) e modello LLM effettivamente usato (prima indicava "api").
- Celle markdown: riferimenti a Ollama sostituiti con vLLM.
- Cella 8: parametro `VLLM_TENSOR_PARALLEL` (2 = usa entrambe le T4).
- Valori di default della cella 10 riportati a una run nuova (`RESUME_FROM_CKPT = False`, nessun checkpoint precedente).
- README: numero di archi del dataset corretto (967.632, non ~1,2 milioni), descrizione di rewiring, selezione dei seed e metriche, piano delle sessioni, risultati obsoleti rimossi.

---

# Terza revisione — installazione di vLLM

- **Cella 0:** versioni fissate (`vllm==0.30.0`, `transformers==5.18.0`, `huggingface_hub==1.33.0`, `tokenizers==0.23.2`), le stesse delle sessioni funzionanti del 4 ottobre 2026. Con `pip install vllm` senza versione veniva installato `huggingface_hub` 2.1.1, incompatibile, e vLLM non partiva.
- **Cella 9:** se vLLM non risponde il notebook si ferma con un errore, invece di proseguire senza LLM.
- **Orchestratore:** se in uno step falliscono più del 50% delle chiamate LLM (`simulation.max_llm_failure_rate`), lo step si interrompe prima di modificare lo stato; l'ultimo checkpoint resta valido. Prima gli agenti senza risposta restavano fermi e lo step veniva registrato come se nulla fosse successo.
