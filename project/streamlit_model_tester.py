from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.before_after_eval import PAIRING_CHOICES, evaluate_before_after_from_maps  # noqa: E402
from project.inference_utils import (  # noqa: E402
    class_distribution,
    compute_seg_metrics,
    default_palette,
    load_model_bundle,
    map_mask_values,
    mask_to_color,
    overlay_mask,
    predict_mask,
)


DEFAULT_CONFIG = ROOT / "project" / "scratch_hardfix.yaml"
DEFAULT_CKPT = ROOT / "project" / "outputs" / "scratch_hardfix" / "checkpoints" / "best.pth"
PRED_ONLY_WARN_THRESHOLD = 0.01


def _inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
  --bg-a: #081226;
  --bg-b: #11243b;
  --ink-strong: #f3f7ff;
  --ink-mid: #d7e1f0;
  --ink-soft: #a7b8cf;
  --panel: rgba(12, 23, 41, 0.78);
  --panel-strong: rgba(17, 31, 56, 0.94);
  --line: rgba(177, 205, 250, 0.22);
  --teal: #2ed3b7;
  --amber: #ffb566;
  --rose: #ff8d94;
}

html, body, [class*="st-"], [data-testid="stAppViewContainer"] {
  font-family: "Plus Jakarta Sans", sans-serif;
  color: var(--ink-strong);
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 8% 8%, rgba(46, 211, 183, 0.18), transparent 30%),
    radial-gradient(circle at 88% 6%, rgba(255, 181, 102, 0.16), transparent 28%),
    linear-gradient(160deg, var(--bg-a) 0%, var(--bg-b) 100%);
}

[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(7, 14, 28, 0.96) 0%, rgba(11, 20, 38, 0.96) 100%);
  border-right: 1px solid rgba(180, 203, 243, 0.14);
}

[data-testid="stSidebar"] * {
  color: var(--ink-mid) !important;
}

h1, h2, h3 {
  font-family: "Outfit", sans-serif;
  letter-spacing: -0.02em;
}

.hero {
  background:
    linear-gradient(120deg, rgba(16, 28, 48, 0.96), rgba(22, 42, 70, 0.92));
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 1.25rem 1.4rem;
  box-shadow: 0 22px 42px rgba(0, 0, 0, 0.35);
  margin-bottom: 0.95rem;
}

.hero h1 {
  margin: 0;
  font-size: 2rem;
  color: #f8fbff;
}

.hero p {
  margin: 0.35rem 0 0;
  color: var(--ink-mid);
}

.status-chip {
  display: inline-block;
  border-radius: 999px;
  border: 1px solid var(--line);
  padding: 0.15rem 0.62rem;
  font-size: 0.72rem;
  color: var(--ink-mid);
  background: rgba(255, 255, 255, 0.04);
}

.metric-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 0.75rem 0.85rem;
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.24);
}

.metric-label {
  text-transform: uppercase;
  font-size: 0.73rem;
  letter-spacing: 0.09em;
  color: var(--ink-soft);
}

.metric-value {
  font-family: "Outfit", sans-serif;
  font-size: 1.62rem;
  font-weight: 700;
}

[data-baseweb="tab-list"] {
  gap: 0.6rem;
}

button[kind="primary"] {
  background: linear-gradient(90deg, #1ac8b2, #2b8fff) !important;
  color: #031025 !important;
  font-weight: 700 !important;
  border: none !important;
}

button[kind="secondary"] {
  border: 1px solid var(--line) !important;
}

[data-testid="stFileUploaderDropzone"] {
  background: var(--panel-strong);
  border: 1px dashed rgba(202, 221, 252, 0.36);
}

[data-testid="stFileUploaderDropzone"] * {
  color: var(--ink-mid) !important;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
  background: rgba(7, 14, 27, 0.9) !important;
  color: #eaf2ff !important;
  border: 1px solid rgba(177, 205, 250, 0.25) !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 12px;
}
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _load_bundle(config_path: str, checkpoint_path: str, device: str):
    return load_model_bundle(config_path=config_path, checkpoint_path=checkpoint_path, device=device)


def _class_names(cfg: dict) -> list[str]:
    names = cfg["data"].get("class_names")
    num_classes = int(cfg["data"]["num_classes"])
    if isinstance(names, list) and len(names) == num_classes:
        return [str(x) for x in names]
    return [f"class_{i}" for i in range(num_classes)]


def _metric_card(label: str, value: float, tone: str = "var(--teal)") -> None:
    st.markdown(
        f"""
<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value" style="color:{tone};">{value:.4f}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _pred_only_names(metrics: dict, class_names: list[str]) -> list[str]:
    gt = np.asarray(metrics["gt_dist"], dtype=np.float64)
    pred = np.asarray(metrics["pred_dist"], dtype=np.float64)
    idx = np.where((gt <= 1e-12) & (pred >= PRED_ONLY_WARN_THRESHOLD))[0].tolist()
    return [class_names[i] for i in idx]


def _read_upload_rgb(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not decode image: {uploaded_file.name}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_upload_mask(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ValueError(f"Could not decode mask: {uploaded_file.name}")
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def _persist_uploads(files, target_dir: Path) -> dict[str, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for f in files:
        p = target_dir / f.name
        p.write_bytes(f.getvalue())
        out[p.stem] = p
    return out


def _render_single_image_tab(config_path: str, checkpoint_path: str, device: str) -> None:
    st.subheader("Single Image Inference")
    st.caption("Upload one image and optionally a GT mask to compute mIoU / mAP50 / Dice / Pixel Accuracy.")

    col_up_1, col_up_2 = st.columns(2)
    with col_up_1:
        image_file = st.file_uploader(
            "Image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            key="single_image",
        )
    with col_up_2:
        gt_file = st.file_uploader(
            "Optional GT mask",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            key="single_gt",
            help="Mask values must match raw class ids expected by the config label_map.",
        )

    run = st.button("Run Inference", type="primary", use_container_width=True, key="run_single")

    if not run:
        return
    if image_file is None:
        st.warning("Upload an image first.")
        return

    try:
        with st.spinner("Running model..."):
            bundle = _load_bundle(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
            cfg = bundle["cfg"]
            class_names = _class_names(cfg)
            num_classes = int(cfg["data"]["num_classes"])
            palette = default_palette(num_classes)

            image_rgb = _read_upload_rgb(image_file)
            pred_mask, _ = predict_mask(bundle, image_rgb)
            pred_color = mask_to_color(pred_mask, num_classes=num_classes, palette=palette)
            pred_overlay = overlay_mask(image_rgb, pred_color, alpha=0.56)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return

    c1, c2, c3 = st.columns(3)
    c1.image(image_rgb, caption="Input", use_container_width=True)
    c2.image(pred_color, caption="Predicted mask", use_container_width=True)
    c3.image(pred_overlay, caption="Overlay", use_container_width=True)

    png_mask = pred_mask.astype(np.uint16 if num_classes > 255 else np.uint8)
    pred_png = cv2.imencode(".png", png_mask)[1].tobytes()
    st.download_button(
        "Download predicted mask (class-index PNG)",
        data=pred_png,
        file_name=f"{Path(image_file.name).stem}_pred.png",
        mime="image/png",
    )

    pred_dist = class_distribution(pred_mask, num_classes=num_classes)
    dist_df = pd.DataFrame({"class": class_names, "pred_dist": pred_dist})

    if gt_file is not None:
        gt_raw = _read_upload_mask(gt_file)
        gt_mask = map_mask_values(gt_raw, cfg)
        gt_color = mask_to_color(gt_mask, num_classes=num_classes, palette=palette)
        gt_overlay = overlay_mask(image_rgb, gt_color, alpha=0.56)
        metrics = compute_seg_metrics(pred_mask=pred_mask, gt_mask_raw=gt_raw, cfg=cfg)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            _metric_card("mIoU", float(metrics["miou"]), tone="var(--teal)")
        with m2:
            _metric_card("mAP50", float(metrics["map50"]), tone="var(--amber)")
        with m3:
            _metric_card("Dice", float(metrics["dice"]), tone="#73b9ff")
        with m4:
            _metric_card("Pixel Acc", float(metrics["pixel_acc"]), tone="#b8d5ff")

        gt_valid = gt_mask[gt_mask != int(cfg["data"]["ignore_index"])]
        if gt_valid.size > 0:
            gt_counts = np.bincount(gt_valid.reshape(-1), minlength=num_classes).astype(np.float64)
            dist_df["gt_dist"] = (gt_counts / max(1.0, gt_counts.sum())).tolist()
        else:
            dist_df["gt_dist"] = [0.0] * num_classes

        st.markdown('<span class="status-chip">Ground Truth Comparison</span>', unsafe_allow_html=True)
        gx, gy = st.columns(2)
        gx.image(gt_color, caption="GT mask", use_container_width=True)
        gy.image(gt_overlay, caption="GT overlay", use_container_width=True)

        iou_df = pd.DataFrame({"class": class_names, "iou": [float(v) for v in metrics["per_class_iou"]]})
        st.subheader("Per-class IoU")
        st.dataframe(iou_df, use_container_width=True, hide_index=True)

        pred_only = _pred_only_names(metrics, class_names)
        if pred_only:
            st.warning("Predicted classes absent in GT: " + ", ".join(pred_only))

    st.subheader("Class Distribution")
    st.bar_chart(dist_df.set_index("class"), use_container_width=True)


def _render_before_after_tab(config_path: str, checkpoint_path: str, device: str) -> None:
    st.subheader("Before vs After Batch Evaluation")
    st.caption(
        "Upload three sets: before images, after images, and GT masks. Files are paired by matching filename stem."
    )

    u1, u2, u3 = st.columns(3)
    with u1:
        before_files = st.file_uploader(
            "Before images",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="before_files",
        )
    with u2:
        after_files = st.file_uploader(
            "After images",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="after_files",
        )
    with u3:
        gt_files = st.file_uploader(
            "Ground-truth masks",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="gt_files",
        )

    pair_strategy = st.selectbox(
        "Pairing method",
        options=list(PAIRING_CHOICES),
        index=list(PAIRING_CHOICES).index("auto"),
        help=(
            "auto = stem -> normalized name -> numeric id -> perceptual hash -> index fallback. "
            "Use hash/index when filenames are unrelated."
        ),
    )

    run_eval = st.button("Run Before/After Evaluation", type="primary", use_container_width=True, key="run_batch")
    if run_eval:
        if not before_files or not after_files or not gt_files:
            st.warning("Upload all three sets before running.")
        else:
            try:
                with st.spinner("Scoring before and after sets..."):
                    bundle = _load_bundle(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
                    with tempfile.TemporaryDirectory(prefix="before_after_eval_") as tmp:
                        tmp_root = Path(tmp)
                        before_map = _persist_uploads(before_files, tmp_root / "before")
                        after_map = _persist_uploads(after_files, tmp_root / "after")
                        gt_map = _persist_uploads(gt_files, tmp_root / "gt")

                        results = evaluate_before_after_from_maps(
                            bundle=bundle,
                            before_map=before_map,
                            after_map=after_map,
                            gt_map=gt_map,
                            save_panels=False,
                            pair_strategy=pair_strategy,
                        )
                    st.session_state["before_after_results"] = results
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

    results = st.session_state.get("before_after_results")
    if not results:
        return

    agg = results["aggregate"]
    before = agg["before"]
    after = agg["after"]

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        _metric_card("Before mIoU", float(before["miou"]), tone="var(--rose)")
    with k2:
        _metric_card("After mIoU", float(after["miou"]), tone="var(--teal)")
    with k3:
        _metric_card("mIoU Gain", float(agg["miou_gain"]), tone="#76d5ff")
    with k4:
        _metric_card("Before mAP50", float(before["map50"]), tone="var(--rose)")
    with k5:
        _metric_card("After mAP50", float(after["map50"]), tone="var(--amber)")
    with k6:
        _metric_card("mAP50 Gain", float(agg["map50_gain"]), tone="#f8d27b")

    st.markdown('<span class="status-chip">Pair Counts</span>', unsafe_allow_html=True)
    st.write(results["counts"])
    if "pairing" in results:
        with st.expander("Pairing Diagnostics"):
            st.json(results["pairing"])

    class_names = agg.get("class_names") or [f"class_{i}" for i in range(results["num_classes"])]
    iou_df = pd.DataFrame(
        {
            "class": class_names,
            "before_iou": before["per_class_iou"],
            "after_iou": after["per_class_iou"],
        }
    )
    st.subheader("Per-class IoU Comparison")
    st.bar_chart(iou_df.set_index("class"), use_container_width=True)

    per_image = pd.DataFrame(results["per_image"])
    if not per_image.empty:
        before_metrics = pd.json_normalize(per_image["before"])
        after_metrics = pd.json_normalize(per_image["after"])
        before_metrics.columns = [f"before_{c}" for c in before_metrics.columns]
        after_metrics.columns = [f"after_{c}" for c in after_metrics.columns]
        per_image = pd.concat([per_image.drop(columns=["before", "after"]), before_metrics, after_metrics], axis=1)
        per_image = per_image.sort_values("map50_gain", ascending=False)

    st.subheader("Per-image Breakdown")
    st.dataframe(per_image, use_container_width=True, hide_index=True)

    dist_df = pd.DataFrame(
        {
            "class": class_names,
            "gt_before": before["gt_dist"],
            "pred_before": before["pred_dist"],
            "gt_after": after["gt_dist"],
            "pred_after": after["pred_dist"],
        }
    )
    st.subheader("Distribution Alignment")
    st.dataframe(dist_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download results.json",
        data=json.dumps(results, indent=2).encode("utf-8"),
        file_name="before_after_results.json",
        mime="application/json",
    )


def main() -> None:
    st.set_page_config(page_title="Segmentation QA Studio", layout="wide", initial_sidebar_state="expanded")
    _inject_styles()

    st.markdown(
        """
<div class="hero">
  <h1>Segmentation QA Studio</h1>
  <p>One app for live inference and before/after quality benchmarking with mIoU and mAP50.</p>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Model Setup")
        config_path = st.text_input("Config path", value=str(DEFAULT_CONFIG))
        checkpoint_path = st.text_input("Checkpoint path", value=str(DEFAULT_CKPT))
        device = st.selectbox("Device", options=["cuda", "cpu"], index=0)

        if Path(config_path).exists():
            st.success("Config found")
        else:
            st.error("Config not found")
        if Path(checkpoint_path).exists():
            st.success("Checkpoint found")
        else:
            st.error("Checkpoint not found")

    if not Path(config_path).exists() or not Path(checkpoint_path).exists():
        st.warning("Fix config/checkpoint path in sidebar before running inference.")

    tab_single, tab_batch = st.tabs(["Single Image QA", "Before vs After QA"])

    with tab_single:
        _render_single_image_tab(config_path=config_path, checkpoint_path=checkpoint_path, device=device)

    with tab_batch:
        _render_before_after_tab(config_path=config_path, checkpoint_path=checkpoint_path, device=device)


if __name__ == "__main__":
    main()
