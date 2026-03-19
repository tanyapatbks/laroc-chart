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
- Phase 1 dataset inventory, validation, preparation, training, and inference entrypoints
- Phase 3 segmentation inventory, preparation, training, and inference entrypoints
- deterministic category mapping
- a reusable feature extraction module
- phase wrappers for detection/segmentation/model inference
- a rewritten Phase 2 calibration pipeline with quality metrics and curated evaluation support

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
coral-thesis phase1-inventory --config configs/base.yaml
coral-thesis phase1-prepare --config configs/base.yaml
```

## Phase 1 Workflow

Phase 1 rebuilds chart detection in four explicit steps:

1. Inventory
   Inspect the dataset, count images/labels, and flag missing or invalid annotations.
2. Preparation
   Build a clean YOLO dataset split under `artifacts/outputs/phase1/dataset`.
   V2 supports both symlink-based and materialized datasets; the active mode is controlled from config.
3. Training
   Train the detector from the configured YOLO backbone.
   V2 prewarms font lookup and image/label verification first so the initial run does not appear to hang.
   Samples that exceed the configured verification timeout can be dropped automatically before YOLO training starts.
4. Inference
   Run chart detection and crop extraction using trained weights.

Example commands:

```bash
coral-thesis phase1-inventory --config configs/base.yaml
coral-thesis phase1-prepare --config configs/base.yaml
coral-thesis phase1-train --config configs/base.yaml
coral-thesis phase1-train --config configs/base.yaml --epochs 1 --image-size 320 --batch-size 4 --run-name chart_detector_smoke
coral-thesis phase1-infer --config configs/base.yaml --source ../dataset/P8250005.JPG
coral-thesis phase1-infer --config configs/base.yaml --source ../dataset/P8250005.JPG --weights artifacts/outputs/phase1/training/chart_detector_smoke/weights/best.pt
coral-thesis phase2-baseline --config configs/base.yaml
coral-thesis phase2-calibrate --config configs/base.yaml --source-image ../dataset/P8250008.JPG --chart-crop ../runs/inference/crops/P8250008_chart_0.jpg --output-name sample
coral-thesis phase2-calibrate-batch --config configs/base.yaml --source-dir ../dataset --crops-dir ../runs/inference/crops --crop-glob 'P8250008_chart_0.jpg' --report-name sample_batch
coral-thesis phase3-inventory --config configs/base.yaml
coral-thesis phase3-prepare --config configs/base.yaml
```

After `phase1-train`, `phase1-infer` will automatically look for
`artifacts/outputs/phase1/training/chart_detector/weights/best.pt`
if `models.chart_detector_weights` is still `null`.

## Phase 2 Workflow

Phase 2 rebuilds color calibration around explicit chart patch sampling:

1. Analyze the baseline chart into a reusable patch profile.
2. Sample the detected chart crop using the same grid geometry.
3. Normalize the chart crop before sampling so perspective/skew affects the calibration less.
4. Fit a linear color transform from crop colors to baseline colors.
5. Apply that transform to the full source image.
6. Record before/after patch error metrics so every calibration run is measurable.

Example commands:

```bash
coral-thesis phase2-baseline --config configs/base.yaml
coral-thesis phase2-calibrate --config configs/base.yaml --source-image ../dataset/P8250008.JPG --chart-crop ../runs/inference/crops/P8250008_chart_0.jpg --output-name sample
coral-thesis phase2-calibrate-batch --config configs/base.yaml --source-dir ../dataset --crops-dir ../runs/inference/crops --crop-glob 'P8250008_chart_0.jpg' --report-name sample_batch
coral-thesis phase2-evaluate-manifest --config configs/base.yaml --report-name phase2_baseline
```

Each Phase 2 run now saves:

- calibrated output images
- normalized chart views used for patch sampling
- per-image quality metrics (`before`, `after`, deltas, improvement ratio)
- batch-level summaries for calibration quality, normalization strategy counts, unreadable inputs, and timeout failures

Batch calibration is now fail-fast per sample through `phase2.sample_timeout_seconds`, so one unstable image does not stall the whole run.

Curated evaluation now lives in `configs/phase2_evaluation_manifest.yaml`.
The manifest is intended to hold a small, stable mix of known-good and known-bad cases so Phase 2 can be regression-checked before Phase 3 work begins.
Evaluation reports are written to `artifacts/reports/phase2`.

## Phase 3 Workflow

Phase 3 starts by cleaning up segmentation training data before any new model work:

1. Audit segmentation images from `../dataset` and segmentation labels from `../data/phase3/labels`.
2. Compare those labels against the legacy split under `../data/phase3`.
3. Deduplicate noisy legacy aliases such as `P8250006 2.JPG`.
4. Normalize mixed legacy annotations so 5-token bbox rows are converted into polygon masks during dataset preparation.
5. Rebuild a clean YOLO segmentation dataset under `artifacts/outputs/phase3/dataset`.
6. Train a clean segmentation model from the rebuilt dataset.
7. Run segmentation inference from trained weights.

Example commands:

```bash
coral-thesis phase3-inventory --config configs/base.yaml
coral-thesis phase3-prepare --config configs/base.yaml
coral-thesis phase3-train --config configs/base.yaml
coral-thesis phase3-train --config configs/base.yaml --epochs 1 --image-size 320 --batch-size 4 --run-name coral_segmenter_smoke
coral-thesis phase3-infer --config configs/base.yaml --source ../dataset/P8250008.JPG
coral-thesis phase3-infer --config configs/base.yaml --source ../dataset/P8250008.JPG --weights artifacts/outputs/phase3/training/coral_segmenter_smoke/weights/best.pt
```

## Next Build Steps

1. Rewrite Phase 2 color calibration with explicit chart patch correspondence.
2. Add dataset manifest generation and label validation for later phases.
3. Add training and evaluation entrypoints per remaining phase.
4. Add experiment tracking and metrics reporting.
