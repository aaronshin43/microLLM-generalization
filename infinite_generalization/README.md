# Infinite Generalization

This directory contains the code and documents for the infinite-length generalization experiments.

The documents in `documents/` are the source of truth for the research plan and task definition.

## Environment

Create and activate the virtual environment from the repository root:

```powershell
Set-Location D:\03_Coding\microLLM-generalization
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .\infinite_generalization
```

For CUDA-enabled PyTorch, activate the same environment and install the CUDA wheel:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Run experiment commands from `infinite_generalization/` after activating `.venv`.

## Configuration

Tracked example configs live in `configs/`. Copy one into `configs/local/` for local runs:

```powershell
Set-Location infinite_generalization
Copy-Item configs/stage1_transformer.example.yaml configs/local/stage1_transformer.yaml
```

Local configs under `configs/local/` are ignored by git, except for `.gitkeep`.
Config precedence is: dataclass defaults, then YAML values, then explicit CLI arguments.
YAML files have `task` and `stage` sections. Use `task.eval_lengths` for length sweeps and
`stage.device` for device selection. Supported device values are `auto`, `cpu`, and `cuda`.

## Stage 0: Max-Pooling Baseline

Run a quick smoke test:

```powershell
Set-Location D:\03_Coding\microLLM-generalization
.\.venv\Scripts\Activate.ps1
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage0_baseline --smoke-test
```

Run unit tests:

```powershell
Set-Location D:\03_Coding\microLLM-generalization
.\.venv\Scripts\Activate.ps1
Set-Location infinite_generalization
python -m unittest discover -s tests
```

Run the default Stage 0 experiment:

```powershell
Set-Location D:\03_Coding\microLLM-generalization
.\.venv\Scripts\Activate.ps1
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage0_baseline
```

Run with a config file:

```powershell
Set-Location D:\03_Coding\microLLM-generalization
.\.venv\Scripts\Activate.ps1
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage0_baseline --config configs/local/stage0_baseline.yaml --device cpu
```

Save audit examples with model outputs:

```powershell
Set-Location D:\03_Coding\microLLM-generalization
.\.venv\Scripts\Activate.ps1
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage0_baseline --save-examples
```

Outputs are written to `runs/stage0_maxpool_baseline/`, including `diagnostic_slices_by_length.csv`.
Diagnostic slices use `--diagnostic-examples 2000` by default.

## Stage 1: Minimal Transformer

Run a quick smoke test:

```powershell
Set-Location D:\03_Coding\microLLM-generalization
.\.venv\Scripts\Activate.ps1
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage1_transformer --smoke-test
```

Run the default Stage 1 experiment:

```powershell
Set-Location D:\03_Coding\microLLM-generalization
.\.venv\Scripts\Activate.ps1
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage1_transformer
```

Run with a config file:

```powershell
Set-Location D:\03_Coding\microLLM-generalization
.\.venv\Scripts\Activate.ps1
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage1_transformer --config configs/local/stage1_transformer.yaml --device cuda
```

Save attention summaries for selected diagnostic-slice examples:

```powershell
Set-Location D:\03_Coding\microLLM-generalization
.\.venv\Scripts\Activate.ps1
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage1_transformer --save-attention --save-examples
```

Outputs are written to `runs/stage1_transformer_maxpool/`, including `diagnostic_slices_by_length.csv`.
Diagnostic slices use `--diagnostic-examples 2000` by default.

## Stage 2A: Multi-Length Transformer

Run a quick smoke test:

```powershell
Set-Location D:\03_Coding\microLLM-generalization
.\.venv\Scripts\Activate.ps1
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage2a_transformer_multilength --smoke-test
```

Run the default Stage 2A experiment:

```powershell
Set-Location D:\03_Coding\microLLM-generalization
.\.venv\Scripts\Activate.ps1
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage2a_transformer_multilength --config configs/stage2a_transformer_multilength.example.yaml
```

Stage 2A keeps the Stage 1 transformer architecture fixed and trains on single-length
batches from `stage.train_lengths`, defaulting to `[10, 20, 50, 100]`.
