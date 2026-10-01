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
