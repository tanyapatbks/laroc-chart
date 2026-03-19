# Architecture

## Design Rules

- The legacy project remains read-only reference material.
- Every pipeline phase has a typed contract for its inputs and outputs.
- Configuration is loaded from YAML, validated once, and passed downward.
- Artifacts are written only inside the V2 workspace.
- Deterministic logic should stay separate from model inference wrappers.

## Layers

1. `config.py`
   Loads YAML into typed dataclasses and resolves paths relative to the V2 project root.
2. `domain/`
   Contains the data contracts passed between phases.
3. `phases/`
   Contains each pipeline phase as an isolated unit.
4. `pipeline.py`
   Orchestrates discovery, validation, and phase ordering.
5. `cli.py`
   Exposes project actions through a stable command-line interface.

## Delivery Strategy

1. Foundation
   Clean package structure, config system, tests, and artifact boundaries.
2. Deterministic Phases
   Finalize category mapping, feature extraction, and dataset validation.
3. Vision Phases
   Implement chart detection, color calibration, and coral segmentation cleanly.
4. Modeling
   Add health estimation training, evaluation, and model registry handling.
5. Research Workflow
   Add experiment reporting and reproducibility controls.

## Phase 1 Scope

Phase 1 in V2 is split into distinct responsibilities:

- inventory: inspect the raw dataset and validate YOLO chart annotations
- preparation: create deterministic train/val splits and write a fresh `data.yaml`
- training: run YOLO training from the configured detection backbone
- inference: run detection and export chart crops plus visualization artifacts

This keeps data quality checks separate from model execution and makes the chart detection phase reproducible.
