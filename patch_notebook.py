import json

notebook_path = "notebooks/kaggle_full_run.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        # Patch Ollama installation cell
        if "apt-get install -y -qq zstd pciutils" in source:
            cell["source"] = [
                "# --- Installazione vLLM ---\n",
                "!pip install vllm\n"
            ]
            
        # Patch Ollama server start cell
        elif "subprocess.Popen(" in source and "ollama" in source and "serve" in source:
            cell["source"] = [
                "# --- Avvio server vLLM (OpenAI Compatible) ---\n",
                "import subprocess, os, time\n",
                "\n",
                "# Usiamo il modello AWQ quantizzato per farlo stare comodo nei 15GB della singola T4 di Kaggle,\n",
                "# oppure usiamo tensor parallel su 2 GPU.\n",
                "MODEL_NAME = \"casperhansen/llama-3-8b-instruct-awq\" # 4-bit quantizzato, veloce e leggero\n",
                "\n",
                "env = os.environ.copy()\n",
                "subprocess.Popen(\n",
                "    ['python', '-m', 'vllm.entrypoints.openai.api_server',\n",
                "     '--model', MODEL_NAME,\n",
                "     '--quantization', 'awq',\n",
                "     '--dtype', 'half',\n",
                "     '--max-model-len', '4096',\n",
                "     '--port', '8000',\n",
                "     '--gpu-memory-utilization', '0.90'],\n",
                "    stdout=open('vllm.log', 'w'),\n",
                "    stderr=subprocess.STDOUT,\n",
                "    env=env,\n",
                ")\n",
                "print(\"⏳ Avvio vLLM in background (potrebbe richiedere 1-3 minuti)...\\n\")\n",
                "time.sleep(10) # Pausa iniziale\n"
            ]
            
        # Patch Healthcheck cell
        elif "http://localhost:11434/v1/models" in source:
            cell["source"] = [
                "import requests, time\n",
                "\n",
                "# Healthcheck vLLM\n",
                "ok = False\n",
                "for i in range(40):\n",
                "    try:\n",
                "        response = requests.get('http://localhost:8000/v1/models', timeout=2)\n",
                "        if response.status_code == 200:\n",
                "            ok = True\n",
                "            break\n",
                "    except Exception:\n",
                "        pass\n",
                "    time.sleep(5)\n",
                "\n",
                "if ok:\n",
                "    models = [m['id'] for m in response.json().get('data', [])]\n",
                "    print(f'✅ vLLM è online! Modelli: {models}')\n",
                "else:\n",
                "    print('❌ Errore: vLLM non sta rispondendo')\n",
                "    !tail -n 30 vllm.log\n"
            ]

        # Patch Pipeline parameters cell
        elif "cfg.llm.backend = \"api\"" in source:
            # We want to change the overrides
            new_source = []
            for line in cell["source"]:
                if "cfg.llm.backend" in line:
                    new_source.append("cfg.llm.backend = \"local\"\n")
                elif "cfg.llm.max_concurrent_requests" in line:
                    new_source.append("cfg.llm.max_concurrent_requests = 150  # vLLM fa il batching!\n")
                elif "cfg.llm.local.model" in line:
                    new_source.append("cfg.llm.local.model = \"casperhansen/llama-3-8b-instruct-awq\"\n")
                    new_source.append("cfg.llm.local.base_url = \"http://localhost:8000/v1\"\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source

    elif cell["cell_type"] == "markdown":
        source = "".join(cell["source"])
        if "Setup Ollama" in source:
            cell["source"] = [
                "## 0.1 Setup vLLM (LLM Locale ad altissime prestazioni)\n",
                "Sostituiamo Ollama con vLLM per sfruttare il **Continuous Batching**. Questo abbatterà i tempi da 12 ore a pochi minuti."
            ]


with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook patched successfully.")
