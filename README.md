# Anti-Fragile Semantic Segmentation for Off-Road Autonomy

This repo scaffolds an anti-fragile semantic segmentation system built to generalize under domain shift in desert environments using synthetic data from Falcon (Duality AI).

## What This Implements
- Domain Randomization: photometric + atmospheric augmentations
- Consistency Learning: stable predictions across transformations
- Hard Example Mining: focus on difficult pixels/classes
- Test-Time Adaptation: entropy minimization on novel domains

## Project Structure
```text
configs/
  default.yaml
data/
  synthetic/
    images/
    masks/
  synthetic_val/
    images/
    masks/
  novel/
    images/
    masks/
outputs/
  checkpoints/
  logs/
scripts/
  train.py
  evaluate.py
  tta.py
src/
  data/
    datasets.py
    transforms.py
  models/
    factory.py
    unet.py
  training/
    consistency.py
    hard_mining.py
    losses.py
    metrics.py
    trainer.py
  tta/
    adapt.py
  utils/
    config.py
    logging.py
    seed.py
```

## Install
```bash
pip install -r requirements.txt
```
Note: `cv2` is the import name; the pip package is `opencv-python-headless` for headless server environments.

## Quickstart
```bash
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pth
python scripts/tta.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pth
```

## High-Performance DeepLabV3+ Preset
Use the production-focused config for stronger quality/cost balance:

```bash
python scripts/train.py --config configs/prod_deeplabv3plus.yaml
python scripts/evaluate.py --config configs/prod_deeplabv3plus.yaml --checkpoint outputs/deeplabv3plus_prod/checkpoints/best.pth --use-ema
```

Resume training from `last.pth`:

```bash
python scripts/train.py --config configs/prod_deeplabv3plus.yaml --resume outputs/deeplabv3plus_prod/checkpoints/last.pth
```

## DeepLabV3+ ResNet-50 (Production)
Dedicated ResNet-50 config:

```bash
python scripts/train.py --config configs/deeplabv3plus_resnet50.yaml
python scripts/evaluate.py --config configs/deeplabv3plus_resnet50.yaml --checkpoint outputs/deeplabv3plus_resnet50/checkpoints/best.pth --use-ema
```

## Automated Sweep (DeepLabV3+ ResNet-50)
Run hyperparameter sweep until target metrics are met or trials are exhausted:

```bash
python scripts/sweep_deeplab_resnet50.py \
  --base-config configs/deeplabv3plus_resnet50.yaml \
  --target-miou 0.8 \
  --target-map50 0.8 \
  --num-epochs 60 \
  --max-trials 24
```

Sweep outputs are written under `outputs/sweeps/...` with `summary.json` and `summary.csv`.

## ONNX Export + Cost Benchmark
Install optional ONNX dependencies:

```bash
pip install -r requirements-onnx.txt
```

Export checkpoint to ONNX:

```bash
python scripts/export_onnx.py \
  --config configs/deeplabv3plus_resnet50.yaml \
  --checkpoint outputs/deeplabv3plus_resnet50/checkpoints/best.pth \
  --output models/deeplabv3plus_resnet50.onnx \
  --use-ema \
  --dynamic
```

Benchmark PyTorch vs ONNX Runtime:

```bash
python scripts/benchmark_inference.py \
  --config configs/deeplabv3plus_resnet50.yaml \
  --checkpoint outputs/deeplabv3plus_resnet50/checkpoints/best.pth \
  --onnx models/deeplabv3plus_resnet50.onnx \
  --device cpu \
  --batch-size 1 \
  --height 384 --width 384 \
  --iters 200 \
  --out-json outputs/benchmarks/deeplabv3plus_resnet50_cpu.json
```

## Data Layout
Images and masks must share filenames (different extensions allowed).

```text
data/
  synthetic/
    images/
      0001.png
    masks/
      0001.png
```

## Notes
- The scaffold is intentionally minimal and modular so you can drop in your own dataset reader, model backbone, and augmentations.
- `default.yaml` is the single place to configure training and evaluation.

## Deploy On Google Cloud Run (GCP)
This repo includes GCP-ready deployment files:
- `Dockerfile`
- `cloudbuild.yaml`
- `.gcloudignore`
- `scripts/deploy_gcp.ps1`
- `configs/deploy.yaml`

### 1) One-time GCP setup
```bash
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### 2) Configure app environment variables
- `APP_CONFIG=configs/deploy.yaml`
- `APP_CHECKPOINT_PATH=models/best.pth`
- `APP_CHECKPOINT_URL=<public direct URL to your trained best.pth>`

If `models/best.pth` does not exist in the container, the app auto-downloads it from `APP_CHECKPOINT_URL`.

### 3) Optional local Docker test
```bash
docker build -t desert-vision .
docker run -p 8501:8501 \
  -e APP_CONFIG=configs/deploy.yaml \
  -e APP_CHECKPOINT_PATH=models/best.pth \
  -e APP_CHECKPOINT_URL="<public direct URL>" \
  desert-vision
```
Open `http://localhost:8501`

### 4) Deploy to Cloud Run
Option A (PowerShell helper):
```powershell
.\scripts\deploy_gcp.ps1 `
  -ProjectId "<YOUR_PROJECT_ID>" `
  -Region "us-central1" `
  -ServiceName "desert-vision" `
  -AppCheckpointUrl "<public direct URL>" `
  -Memory "4Gi" `
  -Cpu "2" `
  -Timeout 600 `
  -MinInstances 0 `
  -MaxInstances 2
```

Option B (direct `gcloud`):
```bash
gcloud builds submit --config cloudbuild.yaml --substitutions=_SERVICE_NAME=desert-vision,_REGION=us-central1,_APP_CONFIG=configs/deploy.yaml,_APP_CHECKPOINT_PATH=models/best.pth,_APP_CHECKPOINT_URL="<public direct URL>",_MEMORY=4Gi,_CPU=2,_TIMEOUT=600,_MIN_INSTANCES=0,_MAX_INSTANCES=2
```
Note: deployment is configured as private-by-default (`--no-allow-unauthenticated`).

### 5) Get deployed URL
```bash
gcloud run services describe desert-vision --region us-central1 --format="value(status.url)"
```
