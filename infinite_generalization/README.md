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

Outputs are written to `runs/stage0_maxpool_baseline/`.
