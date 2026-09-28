# Micro LLM Generalization

This repo contains Aaron Shin's experiments for the research on generalization in "micro" LLMs.

For the reduced binary detection baseline and training-length follow-up, start with
the [detection guide](infinite_generalization/documents/training_length/DETECTION_GUIDE.md)
and [baseline reproduction record](infinite_generalization/documents/training_length/BASELINE_REPRODUCTION.md).
See the [project README](infinite_generalization/README.md) for experiment setup.

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
