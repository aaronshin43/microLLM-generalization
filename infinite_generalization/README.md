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

Use this setup before running tests or experiments:

```powershell
Set-Location D:\03_Coding\microLLM-generalization
.\.venv\Scripts\Activate.ps1
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
```

## Configuration

Tracked example configs live in `configs/`. Copy one into `configs/local/` for local runs:

```powershell
Copy-Item configs/stage1_transformer.example.yaml configs/local/stage1_transformer.yaml
```

Local configs under `configs/local/` are ignored by git, except for `.gitkeep`.
Config precedence is: dataclass defaults, then YAML values, then explicit CLI arguments.
YAML files have `task` and `stage` sections.

Key config fields:

- `task.eval_lengths`: evaluation length sweep.
- `stage.device`: `auto`, `cpu`, or `cuda`.
- `stage.train_length`: single training length for Stage 0 and Stage 1.
- `stage.train_lengths`: multi-length training set for Stage 2A.

## Tests

```powershell
python -m unittest discover -s tests
```

## Stage 0: Max-Pooling Baseline

Smoke test:

```powershell
python -m stage0_baseline --smoke-test --save-examples
```

Default run:

```powershell
python -m stage0_baseline --config configs/stage0_baseline.example.yaml --save-examples
```

Outputs are written to `runs/stage0_maxpool_baseline/`, including `metrics_by_length.csv`,
`diagnostic_slices_by_length.csv`, and optional audit CSVs under `examples/`.

## Stage 1: Minimal Transformer

Smoke test:

```powershell
python -m stage1_transformer --smoke-test --save-examples --save-attention
```

Default run:

```powershell
python -m stage1_transformer --config configs/stage1_transformer.example.yaml --save-examples --save-attention
```

Outputs are written to `runs/stage1_transformer_maxpool/`, including `metrics_by_length.csv`,
`diagnostic_slices_by_length.csv`, audit CSVs under `examples/`, and attention summaries under
`attention/`.

## Stage 2A: Multi-Length Transformer

Smoke test:

```powershell
python -m stage2a_transformer_multilength --smoke-test --save-examples --save-attention
```

Default run:

```powershell
python -m stage2a_transformer_multilength --config configs/stage2a_transformer_multilength.example.yaml --save-examples --save-attention
```

Stage 2A keeps the Stage 1 transformer architecture fixed and trains on single-length batches
from `stage.train_lengths`, defaulting to `[10, 20, 50, 100]`.

## Stage Flags

Shared flags:

- `--config PATH`: load a YAML config.
- `--device {auto,cpu,cuda}`: override device selection.
- `--eval-lengths ...`: override `task.eval_lengths`.
- `--diagnostic-examples N`: set examples per diagnostic slice.
- `--save-examples`: write audit CSVs with example sequences and model outputs.
- `--examples-per-class N`: control audit examples per class.
- `--preview-tokens N`: control long-sequence preview width in audit CSVs.
- `--smoke-test`: run a small fast pipeline check.

Stage 0 flags:

- `--train-length N`: override the single train length.
- `--embedding-dim N`: set embedding width.
- `--hidden-dim N`: set per-token MLP hidden width.

Stage 1 flags:

- `--train-length N`: override the single train length.
- `--d-model N`, `--num-heads N`, `--num-layers N`, `--dim-feedforward N`, `--dropout X`: set transformer shape.
- `--save-attention`: write attention summary CSVs.
- `--save-raw-attention`: also write raw attention tensors.
- `--attention-examples-per-class N`: control attention examples per diagnostic slice.

Stage 2A flags:

- `--train-lengths ...`: override the multi-length training set.
- `--train-examples-per-length N`: set training examples for each train length.
- `--val-examples-per-length N`: set validation examples for each train length.
- `--d-model N`, `--num-heads N`, `--num-layers N`, `--dim-feedforward N`, `--dropout X`: set transformer shape.
- `--save-attention`: write attention summary CSVs.
- `--save-raw-attention`: also write raw attention tensors.
- `--attention-examples-per-class N`: control attention examples per diagnostic slice.
