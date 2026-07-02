import json

with open("notebooks/kaggle_full_run.ipynb", "r") as f:
    nb = json.load(f)

# Fix Cell 5 (Markdown)
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown":
        text = "".join(cell.get("source", []))
        if "Setup Ollama" in text:
            cell["source"] = [
                "## 0.1 Setup vLLM (LLM API Locale)\n",
                "Installa vLLM, avvia l'API server in background distribuito sulle 2 T4 e scarica il modello Llama 3."
            ]
            break

# Fix Cell 8 (Code)
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        text = "".join(cell.get("source", []))
        if "CONFIG_PATH       = 'config.yaml'" in text:
            new_source = []
            for line in cell["source"]:
                if line.startswith("USE_MOCK_LLM      = False"):
                    new_source.append(line)
                    new_source.append("print(f'  USE_MOCK_LLM   = {USE_MOCK_LLM}')\n")
                elif line.startswith("PHASE2_STEPS      = 5"):
                    new_source.append(line)
                    new_source.append("print(f'  PHASE2_STEPS   = {PHASE2_STEPS}')\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source
            break

with open("notebooks/kaggle_full_run.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
