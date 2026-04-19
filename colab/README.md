# Colab Run Guide

Use this project from Google Colab without saving checkpoints.

## 1. Clone the repo

```bash
!git clone https://github.com/ShaownBuetBme/BeamProject.git
%cd BeamProject
```

## 2. Install dependencies

```bash
!pip install -r requirements.txt
```

If `torch` installation needs a CUDA-specific build, install the Colab version instead.

## 3. Run the multimodal baseline

```bash
!python train_multimodal_cli.py --config configs/multimodal_baseline.yaml
```

## 4. Save outputs to Drive if needed

Override the output directory from the CLI:

```bash
!python train_multimodal_cli.py \
  --config configs/multimodal_baseline.yaml \
  --output-dir /content/drive/MyDrive/beam_runs
```

## 5. Reports produced

Each run writes:

- `metrics_summary.json`
- `fold_metrics.csv`
- `all_predictions.csv`
- per-fold `metrics.json`
- per-fold `training_history.csv`
- plots under each fold directory
