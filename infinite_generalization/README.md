# Infinite Generalization

This directory contains the code and documents for the infinite-length generalization experiments.

The documents in `documents/` are the source of truth for the research plan and task definition.

## Stage 0: Max-Pooling Baseline

Run a quick smoke test:

```powershell
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage0_baseline --smoke-test
```

Run unit tests:

```powershell
Set-Location infinite_generalization
python -m unittest discover -s tests
```

Run the default Stage 0 experiment:

```powershell
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage0_baseline
```

Save audit examples with model outputs:

```powershell
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage0_baseline --save-examples
```

Outputs are written to `runs/stage0_maxpool_baseline/`, including `diagnostic_slices_by_length.csv`.
Diagnostic slices use `--diagnostic-examples 2000` by default.

## Stage 1: Minimal Transformer

Run a quick smoke test:

```powershell
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage1_transformer --smoke-test
```

Run the default Stage 1 experiment:

```powershell
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage1_transformer
```

Save attention summaries for selected diagnostic-slice examples:

```powershell
Set-Location infinite_generalization
$env:PYTHONPATH = "src"
python -m stage1_transformer --save-attention --save-examples
```

Outputs are written to `runs/stage1_transformer_maxpool/`, including `diagnostic_slices_by_length.csv`.
Diagnostic slices use `--diagnostic-examples 2000` by default.
