# Anti-Fragile Semantic Segmentation for Off-Road Autonomy

This repository currently implements a scratch semantic segmentation pipeline focused on class imbalance robustness for off-road scenes.

The active, working code path in this snapshot is under `project/`:
- Training: `project/train.py`
- Inference helpers: `project/inference_utils.py`
- Before/after benchmarking: `project/before_after_eval.py`
- Streamlit QA apps: `project/streamlit_model_tester.py`, `project/streamlit_before_after.py`

## Current Repository Layout

```text
project/
  config.py                  # Config defaults, merge, path resolution, validation
  dataset.py                 # Dataset, label mapping, object-focused crop, dataset stats
  sampler.py                 # Class-aware and weighted sampling
  model.py                   # UNet-from-scratch with optional deep supervision
  loss.py                    # Hybrid loss (CE + Dice + Focal + KL distribution alignment)
  train.py                   # End-to-end training loop and validation
  inference_utils.py         # Model loading, prediction, overlays, per-image metrics
  before_after_eval.py       # Batch before/after evaluation with robust pairing
  streamlit_model_tester.py  # Single-image + batch QA dashboard
  streamlit_before_after.py  # Focused before/after dashboard
  scratch_hardfix.yaml       # Full training config
  smoke_hardfix.yaml         # Fast smoke-test config
  outputs/
README.md
requirements.txt
streamlit_app.py             # Legacy UI path (see note below)
main.py                      # Legacy CLI path (see note below)
```

## Step-by-Step Implementation Details

### 1) Configuration bootstrap (`project/config.py`)
1. Load YAML config.
2. Deep-merge it with `DEFAULT_CONFIG`.
3. Resolve all `paths.*` values relative to the config file location.
4. Validate:
   - `data.num_classes > 1`
   - `ignore_index` is valid
   - `model.pretrained` is disabled (hard requirement)
   - train/val image and mask folders exist
   - key training hyperparameters are positive

### 2) Dataset preparation (`project/dataset.py`)
1. Enumerate image files from `paths.train_images` / `paths.val_images`.
2. Align masks by filename stem (image `abc.png` must have mask stem `abc.*`).
3. Convert raw mask values to class indices via optional `label_map`.
4. Compute dataset statistics:
   - per-class pixel counts
   - per-image class sets
   - class-to-image index map
   - rare classes via quantile threshold
5. Training-only logic:
   - object/rare-class focused crop (`object_crop_prob`, `object_crop_size`)
   - Albumentations augmentations (random resized crop, flip, color jitter, blur)
6. Validation logic:
   - deterministic resize + normalize only

### 3) Class-imbalance sampling (`project/sampler.py`)
`train.py` chooses one strategy:
1. `class_aware`:
   - ensures rare-class samples are included each batch
   - fills remaining slots via inverse-frequency class sampling
2. `weighted`:
   - uses image-level weighted random sampling, upweighting rare-class images
3. `random`:
   - standard shuffled minibatches

### 4) Model architecture (`project/model.py`)
1. Build `UNetScratch` (no pretrained backbone).
2. Encoder-decoder UNet blocks with skip connections.
3. Main segmentation head outputs `num_classes` logits.
4. Optional auxiliary head from intermediate decoder feature map for deep supervision.
5. Output is `(main_logits, aux_logits)`.

### 5) Hybrid loss (`project/loss.py`)
Per forward pass, `HybridSegLoss` computes:
1. Weighted Cross-Entropy
2. Multi-class Dice loss
3. Focal loss
4. KL divergence between predicted class distribution and GT distribution

Total loss:
`ce_weight * CE + dice_weight * Dice + focal_weight * Focal + dist_kl_weight * KL`

If auxiliary logits exist, their loss is added with `model.aux_weight`.

### 6) Training and validation loop (`project/train.py`)
1. Set seed and logger.
2. Build train/val datasets and dataloaders.
3. Compute inverse-log class weights from dataset class pixel counts.
4. Create model, `AdamW`, cosine LR scheduler, AMP scaler, and hybrid criterion.
5. For each epoch:
   - train with mixed precision (if CUDA + `amp=true`)
   - apply grad clipping
   - log loss breakdown (`ce`, `dice`, `focal`, `kl`)
   - run validation and compute confusion-matrix metrics
   - log per-class IoU and GT/pred class distributions
   - save `last.pth`, and update `best.pth` when mIoU improves
6. Guardrails:
   - optional assertion that every batch contains at least one rare class
   - optional assertion that all dataset classes are seen each epoch
   - warning when model predicts classes absent from GT above threshold

### 7) Metrics and outputs (`project/utils.py`)
From confusion matrix:
- `miou`
- `dice`
- `map50` (fraction of classes with IoU >= 0.5)
- `pixel_acc`
- `per_class_iou`
- `gt_dist` and `pred_dist`

Training outputs go to:
- `project/outputs/<run_name>/train.log`
- `project/outputs/<run_name>/checkpoints/best.pth`
- `project/outputs/<run_name>/checkpoints/last.pth`

### 8) Before vs after evaluation (`project/before_after_eval.py`)
1. Load config + checkpoint bundle.
2. Build file maps for before/after/GT directories.
3. Pair images robustly (`auto`, `stem`, `normalized`, `numeric`, `hash`, `index`).
4. For each matched triplet:
   - predict mask on before and after image
   - compare each prediction against GT
   - record per-image metrics and gains
5. Aggregate confusion matrices across all pairs.
6. Save `results.json` and optional visualization panels.

### 9) Streamlit QA applications
1. `project/streamlit_model_tester.py`
   - single-image inference with optional GT scoring
   - batch before/after evaluation in one dashboard
2. `project/streamlit_before_after.py`
   - focused before/after QA workflow

Both apps use `inference_utils.py` and `before_after_eval.py`.

## How to Run

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Prepare dataset

For each split, image and mask filenames must match by stem:

```text
train_images/
  0001.png
  0002.png
train_masks/
  0001.png
  0002.png
```

If raw mask ids are not already `[0..num_classes-1]`, define `data.label_map` in YAML.

### 3) Configure paths and hyperparameters
Edit:
- `project/scratch_hardfix.yaml` for full training
- `project/smoke_hardfix.yaml` for quick sanity checks

### 4) Train
```bash
python project/train.py --config project/scratch_hardfix.yaml
```

Quick smoke run:
```bash
python project/train.py --config project/smoke_hardfix.yaml
```

### 5) Run before/after benchmark
```bash
python project/before_after_eval.py ^
  --config project/scratch_hardfix.yaml ^
  --checkpoint project/outputs/scratch_hardfix/checkpoints/best.pth ^
  --before <path-to-before-images> ^
  --after <path-to-after-images> ^
  --gt <path-to-gt-masks> ^
  --output project/outputs/before_after_eval ^
  --pairing auto
```

On Linux/macOS, replace `^` with `\`.

### 6) Launch Streamlit dashboards
```bash
streamlit run project/streamlit_model_tester.py
```
or
```bash
streamlit run project/streamlit_before_after.py
```

## Default Class Mapping in `scratch_hardfix.yaml`

The config maps these raw mask ids into class indices `0..9`:

`[100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000]`

## Troubleshooting

- `Mask missing for image stem ...`:
  Image and mask names are not aligned.
- `Invalid class indices found in mask ...`:
  Update `data.label_map` or clean mask ids.
- `Rare-class coverage failed ...`:
  Lower `sampler.rare_per_batch`, adjust `rare_quantile`, or disable assertion for debugging.
- `Pretrained models are disabled ...`:
  Keep `model.pretrained: false`.



For this branch, use the `project/` scripts documented above.

---

## Test Results on Off-Road Segmentation Test Set

Branch: `test-results-branch`  
Test images: `Offroad_Segmentation_testImages` (Color_Images + Segmentation GT masks)  
Checkpoint: `project/outputs/scratch_hardfix/checkpoints/best.pth`

### How to Run Test Inference

```bash
python project/test_inference.py ^
  --images "C:/Users/ADMIN/Downloads/Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/Color_Images" ^
  --masks  "C:/Users/ADMIN/Downloads/Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/Segmentation" ^
  --config  project/scratch_hardfix.yaml ^
  --checkpoint project/outputs/scratch_hardfix/checkpoints/best.pth ^
  --output  project/outputs/test_results ^
  --device  cpu
```

On Linux/macOS replace `^` with `\`.

Outputs written to `project/outputs/test_results/`:
- `test_results.json` — per-image mIoU, mAP50, Dice, Pixel Accuracy
- `overlays/` — predicted segmentation overlaid on each test image

### Aggregate Metrics (Test Set)

> Evaluated on **1002 test images** using NVIDIA GeForce GTX 1650 (CUDA)

| Metric | Score |
|---|---|
| mIoU | **0.3865** |
| mAP50 | **0.2562** |
| Dice | **0.4892** |
| Pixel Accuracy | **0.5555** |

### Sample Output Visualizations

Each row shows: **Input Image → Predicted Mask → Overlay**

> Add sample overlay images from `project/outputs/test_results/overlays/` here after running inference.
> Example: `![sample](project/outputs/test_results/overlays/0000147.png)`

### Class Legend

| Class Index | Class Name | Color |
|---|---|---|
| 0 | dirt_road | dark gray |
| 1 | gravel | orange |
| 2 | grass | blue |
| 3 | rock | green |
| 4 | water | yellow |
| 5 | mud | purple |
| 6 | sand | teal |
| 7 | vegetation | light orange |
| 8 | obstacle | indigo |
| 9 | sky | brown |
