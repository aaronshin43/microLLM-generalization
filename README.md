# Micro LLM Generalization

This repo contains Aaron Shin's experiments for the research on generalization in "micro" LLMs.

## Environment Setup

Use one virtual environment at the repository root:

```powershell
Set-Location D:\03_Coding\microLLM-generalization
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .\infinite_generalization
```

For CUDA-enabled PyTorch, activate the same environment and replace the default torch wheel:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```
