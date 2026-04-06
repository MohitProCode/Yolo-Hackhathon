from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project.before_after_eval import evaluate_before_after_from_maps  # noqa: E402
from project.inference_utils import load_model_bundle  # noqa: E402


DEFAULT_CONFIG = ROOT / "project" / "scratch_hardfix.yaml"
DEFAULT_CKPT = ROOT / "project" / "outputs" / "scratch_hardfix" / "checkpoints" / "best.pth"


def _inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --text: #173047;
  --line: rgba(23,48,71,0.16);
  --paper: rgba(255,255,255,0.76);
  --teal: #007f7f;
  --amber: #cf5a22;
}

html, body, [class*="st-"], [data-testid="stAppViewContainer"] {
  font-family: "IBM Plex Sans", sans-serif;
  color: var(--text);
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 85% 8%, rgba(207,90,34,0.18), transparent 30%),
    radial-gradient(circle at 12% 7%, rgba(0,127,127,0.18), transparent 30%),
    linear-gradient(180deg, #f3efe6 0%, #fbf9f5 100%);
}

h1, h2, h3 {
  font-family: "Sora", sans-serif;
  letter-spacing: -0.02em;
}

.hero {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1.2rem 1.4rem;
  margin-bottom: 1rem;
  box-shadow: 0 18px 36px rgba(0, 0, 0, 0.08);
}

.kpi-card {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 0.8rem 0.9rem;
  box-shadow: 0 10px 20px rgba(0,0,0,0.06);
}

.kpi-label {
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: rgba(23,48,71,0.65);
}

.kpi-value {
  font-family: "Sora", sans-serif;
  font-weight: 700;
  font-size: 1.6rem;
  color: var(--teal);
}
</style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _load_bundle(config_path: str, checkpoint_path: str, device: str):
    return load_model_bundle(config_path=config_path, checkpoint_path=checkpoint_path, device=device)


def _persist_uploads(files, target_dir: Path) -> dict[str, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for f in files:
        p = target_dir / f.name
        p.write_bytes(f.getvalue())
        out[p.stem] = p
    return out


def _kpi(label: str, value: float, color: str = "#007f7f") -> None:
    st.markdown(
        f"""
<div class="kpi-card">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value" style="color:{color}">{value:.4f}</div>
</div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Before/After Segmentation Evaluator", layout="wide")
    _inject_styles()

    st.markdown(
        """
<div class="hero">
  <h1 style="margin:0;">Before/After Segmentation Evaluator</h1>
  <p style="margin:0.35rem 0 0;color:rgba(23,48,71,0.75);">
    Upload before images, after images, and GT masks. The app scores both sets with your current model and reports mIoU/mAP50 gains.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Model Setup")
        config_path = st.text_input("Config path", value=str(DEFAULT_CONFIG))
        checkpoint_path = st.text_input("Checkpoint path", value=str(DEFAULT_CKPT))
        device = st.selectbox("Device", ["cuda", "cpu"], index=0)
        if not Path(config_path).exists():
            st.error("Config path not found.")
        if not Path(checkpoint_path).exists():
            st.error("Checkpoint path not found.")

    c1, c2, c3 = st.columns(3)
    with c1:
        before_files = st.file_uploader(
            "Before images",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="before_files",
        )
    with c2:
        after_files = st.file_uploader(
            "After images",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="after_files",
        )
    with c3:
        gt_files = st.file_uploader(
            "Ground-truth masks",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            key="gt_files",
            help="Mask filenames should match image filenames by stem.",
        )

    run = st.button("Run Before/After Evaluation", type="primary", use_container_width=True)

    if run:
        if not before_files or not after_files or not gt_files:
            st.warning("Upload all three sets: before, after, and GT masks.")
        elif not Path(config_path).exists() or not Path(checkpoint_path).exists():
            st.warning("Fix config/checkpoint path first.")
        else:
            try:
                with st.spinner("Evaluating..."):
                    bundle = _load_bundle(config_path=config_path, checkpoint_path=checkpoint_path, device=device)
                    with tempfile.TemporaryDirectory(prefix="seg_eval_") as tmp:
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
        _kpi("Before mIoU", float(before["miou"]), "#cf5a22")
    with k2:
        _kpi("After mIoU", float(after["miou"]), "#007f7f")
    with k3:
        _kpi("mIoU Gain", float(agg["miou_gain"]), "#007f7f" if agg["miou_gain"] >= 0 else "#cf5a22")
    with k4:
        _kpi("Before mAP50", float(before["map50"]), "#cf5a22")
    with k5:
        _kpi("After mAP50", float(after["map50"]), "#007f7f")
    with k6:
        _kpi("mAP50 Gain", float(agg["map50_gain"]), "#007f7f" if agg["map50_gain"] >= 0 else "#cf5a22")

    st.subheader("Counts")
    st.write(results["counts"])

    class_names = agg.get("class_names") or [f"class_{i}" for i in range(results["num_classes"])]
    iou_df = pd.DataFrame(
        {
            "class": class_names,
            "before_iou": before["per_class_iou"],
            "after_iou": after["per_class_iou"],
        }
    )
    st.subheader("Per-class IoU")
    st.bar_chart(iou_df.set_index("class"), use_container_width=True)

    per_image = pd.DataFrame(results["per_image"])
    if not per_image.empty:
        before_metrics = pd.json_normalize(per_image["before"])
        after_metrics = pd.json_normalize(per_image["after"])
        before_metrics.columns = [f"before_{c}" for c in before_metrics.columns]
        after_metrics.columns = [f"after_{c}" for c in after_metrics.columns]
        per_image = pd.concat([per_image.drop(columns=["before", "after"]), before_metrics, after_metrics], axis=1)
        per_image = per_image.sort_values("map50_gain", ascending=False)

    st.subheader("Per-image Scores")
    st.dataframe(per_image, use_container_width=True, hide_index=True)

    st.subheader("Distribution Drift")
    dist_df = pd.DataFrame(
        {
            "class": class_names,
            "gt_before": before["gt_dist"],
            "pred_before": before["pred_dist"],
            "gt_after": after["gt_dist"],
            "pred_after": after["pred_dist"],
        }
    )
    st.dataframe(dist_df, use_container_width=True, hide_index=True)

    json_bytes = json.dumps(results, indent=2).encode("utf-8")
    st.download_button(
        "Download results.json",
        data=json_bytes,
        file_name="before_after_results.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
