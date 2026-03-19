# coral-thesis

`coral-thesis` is the clean V2 workspace for the coral health assessment pipeline.
It intentionally starts as a separate project so the legacy prototype can remain untouched and available for reference.

## Goals

- Keep the legacy prototype isolated.
- Rebuild the pipeline with clear boundaries between config, domain objects, phase logic, and orchestration.
- Make the project reproducible, testable, and ready for research iteration.

## Project Layout

```text
coral-thesis/
├── configs/              # YAML configuration files
├── docs/                 # Architecture and design notes
├── artifacts/            # Generated outputs, reports, and temporary files
├── src/coral_thesis/
│   ├── domain/           # Data contracts between phases
│   ├── phases/           # Phase-specific logic
│   ├── cli.py            # Command-line entrypoint
│   ├── config.py         # Typed configuration loading and validation
│   └── pipeline.py       # Pipeline orchestration shell
└── tests/                # Unit tests
```

## Current Scope

This first V2 scaffold provides:

- typed configuration loading
- artifact/workspace bootstrapping
- a clean package layout
- implemented phase contracts
- deterministic category mapping
- a reusable feature extraction module
- phase wrappers for detection/segmentation/model inference

The actual Phase 2 color calibration rewrite is intentionally left as a separate implementation step.

## Data Source Strategy

V2 reuses the existing dataset and baseline chart from the legacy workspace:

- dataset: `../dataset`
- baseline chart: `../baseline_chart.png`

That keeps the new codebase clean while still using the original assets.

## Getting Started

```bash
cd /Users/vebkks/Documents/coral/coral-thesis
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
coral-thesis validate-config --config configs/base.yaml
coral-thesis bootstrap --config configs/base.yaml
coral-thesis describe-pipeline --config configs/base.yaml
```

## Next Build Steps

1. Rewrite Phase 2 color calibration with explicit chart patch correspondence.
2. Add dataset manifest generation and label validation.
3. Add training and evaluation entrypoints per phase.
4. Add experiment tracking and metrics reporting.

