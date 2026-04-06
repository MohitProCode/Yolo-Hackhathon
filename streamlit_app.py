import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Desert Vision Dashboard",
    page_icon="🏜️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0D1117; }

.hero {
    background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #0D1117 100%);
    border: 1px solid #30363D;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero h1 { font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg, #00FFD1, #FF6B35);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
.hero p  { color: #8B949E; font-size: 1rem; margin: 0.4rem 0 0; }

.metric-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #00FFD1; }
.metric-card .label { color: #8B949E; font-size: 0.75rem; text-transform: uppercase;
    letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.metric-card .value { font-size: 2rem; font-weight: 700; color: #00FFD1; }
.metric-card .delta { font-size: 0.85rem; margin-top: 0.2rem; }
.delta-pos { color: #3FB950; }
.delta-neg { color: #F85149; }

.section-title {
    font-size: 1.1rem; font-weight: 600; color: #E6EDF3;
    border-left: 3px solid #00FFD1;
    padding-left: 0.75rem; margin: 1.5rem 0 1rem;
}

.upload-box {
    background: #161B22;
    border: 2px dashed #30363D;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.upload-box:hover { border-color: #00FFD1; }

.run-btn > button {
    background: linear-gradient(90deg, #00FFD1, #00B4D8) !important;
    color: #0D1117 !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.6rem 2rem !important;
    font-size: 1rem !important;
    width: 100% !important;
}

.status-box {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 8px; padding: 1rem;
    font-family: monospace; font-size: 0.8rem;
    color: #3FB950; max-height: 200px; overflow-y: auto;
}

.pair-card {
    background: #161B22; border: 1px solid #30363D;
    border-radius: 10px; padding: 1rem; margin-bottom: 1rem;
}
.pair-header { color: #00FFD1; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem; }

.tag {
    display: inline-block; padding: 0.15rem 0.6rem;
    border-radius: 20px; font-size: 0.72rem; font-weight: 600;
    margin-right: 0.3rem;
}
.tag-before { background: #2D1B1B; color: #F85149; border: 1px solid #F85149; }
.tag-after  { background: #1B2D1B; color: #3FB950; border: 1px solid #3FB950; }
</style>
""", unsafe_allow_html=True)

CONFETTI_JS = """
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.6.0/dist/confetti.browser.min.js"></script>
<script>
(function(){
  var end = Date.now() + 3000;
  var colors = ['#00FFD1','#FF6B35','#3FB950','#58A6FF','#F0883E'];
  (function frame(){
    confetti({particleCount:3, angle:60,  spread:55, origin:{x:0},   colors:colors});
    confetti({particleCount:3, angle:120, spread:55, origin:{x:1},   colors:colors});
    if(Date.now()<end) requestAnimationFrame(frame);
  }());
})();
</script>
"""

CLASS_NAMES = [
    "Road","Low Veg.","Unpaved Rd.","Obstacle",
    "High Veg.","Sky","Vehicle","Rough Trail","Smooth Trail","Water",
]
EPOCH_ID_RE = re.compile(r"Epoch\s+(\d+)")
ROOT = Path(__file__).resolve().parent
DEPLOY_DEFAULT_CONFIG = "deploy.yaml"
DEPLOY_DEFAULT_CHECKPOINT = "models/best.pth"


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path).resolve()


@st.cache_data(show_spinner=False)
def download_checkpoint(url: str, destination: str) -> tuple[bool, str]:
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(url, target)
    except Exception as exc:
        return False, str(exc)
    return True, ""


def set_inference_model_defaults(cfg: dict) -> dict:
    model_cfg = cfg.setdefault("model", {})
    name = str(model_cfg.get("name", "")).lower()
    if name in {"unet", "attention_unet", "attunet", "deeplabv3plus", "deeplabv3+"}:
        if model_cfg.get("pretrained", False):
            model_cfg["use_timm_backbone"] = True
            model_cfg["pretrained"] = False
    return cfg


# ── Helpers ────────────────────────────────────────────────────────────────────
def _parse_metrics(parts):
    m = {}
    for p in parts:
        if "=" not in p: continue
        k, v = p.split("=", 1)
        try: m[k.strip()] = float(v.strip())
        except ValueError: pass
    return m


def parse_log(log_path):
    epochs, last_test = [], {}
    last_val_iou = last_test_iou = None
    section = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Epoch" in line and "| loss=" in line:
            m = EPOCH_ID_RE.search(line)
            if not m: continue
            parts = [p.strip() for p in line.split("|")][1:]
            row = _parse_metrics(parts)
            row["epoch"] = int(m.group(1))
            epochs.append(row)
            section = "val"
        elif line.strip().startswith("Test"):
            last_test = _parse_metrics([p.strip() for p in line.split("|")][1:])
            section = "test"
        elif "Class IoU:" in line:
            try:
                parsed = ast.literal_eval(line.split("Class IoU:")[1].strip())
                if isinstance(parsed, list):
                    if section == "test": last_test_iou = parsed
                    else: last_val_iou = parsed
            except Exception: pass
    return pd.DataFrame(epochs), last_test, last_val_iou, last_test_iou


def save_uploads(files, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        dest = folder / f.name
        dest.write_bytes(f.read())
        paths.append(dest)
    return paths


def colorize_mask(mask_bgr):
    colors = [
        (128,64,128),(107,142,35),(190,153,153),(220,20,60),
        (34,139,34),(70,130,180),(0,0,142),(244,164,96),(255,255,0),(0,191,255),
    ]
    if mask_bgr.ndim == 3:
        gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_bgr
    rgb = np.zeros((*gray.shape, 3), dtype=np.uint8)
    for i, c in enumerate(colors):
        rgb[gray == i] = c[::-1]  # BGR→RGB
    return rgb


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏜️ Desert Vision")
    st.markdown("---")

    configs = sorted((ROOT / "configs").glob("*.yaml"))
    config_names = [c.name for c in configs]
    if not config_names:
        st.error("No config files found in ./configs")
        st.stop()

    env_config_name = Path(os.getenv("APP_CONFIG", DEPLOY_DEFAULT_CONFIG)).name
    if env_config_name in config_names:
        default_config_name = env_config_name
    elif DEPLOY_DEFAULT_CONFIG in config_names:
        default_config_name = DEPLOY_DEFAULT_CONFIG
    elif "deeplabv3plus_resnet50.yaml" in config_names:
        default_config_name = "deeplabv3plus_resnet50.yaml"
    else:
        default_config_name = config_names[0]
    sel_config = st.selectbox("📋 Config", config_names,
                              index=config_names.index(default_config_name))
    config_path = ROOT / "configs" / sel_config

    ckpt_default = os.getenv("APP_CHECKPOINT_PATH", DEPLOY_DEFAULT_CHECKPOINT)
    ckpt_input = st.text_input("🔖 Checkpoint path", ckpt_default)
    ckpt_path = resolve_repo_path(ckpt_input)
    ckpt_download_url = os.getenv("APP_CHECKPOINT_URL", "").strip()
    if not ckpt_path.exists() and ckpt_download_url:
        ok, err = download_checkpoint(ckpt_download_url, str(ckpt_path))
        if ok and ckpt_path.exists():
            st.caption(f"Downloaded checkpoint to `{ckpt_path}`")
        elif err:
            st.warning(f"Checkpoint auto-download failed: {err}")

    st.markdown("---")
    st.markdown("**Team:** The Innovators")
    st.markdown("Aifaz · Mohit · Omkar")
    st.markdown("**Hackathon:** YOLO 🏆")


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏜️ Desert Vision</h1>
  <p>Anti-Fragile Semantic Scene Segmentation · Off-Road Autonomy · YOLO Hackathon</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📈  Training Metrics",
    "🔍  Before vs After Evaluation",
    "🖼️  Segmentation Viewer",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Training Metrics
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    log_dirs = [ROOT / "configs" / "outputs" / "logs", ROOT / "outputs" / "logs"]
    log_files = []
    for d in log_dirs:
        if d.exists():
            log_files.extend(d.glob("run_*.log"))
    log_files = sorted(log_files, key=lambda p: p.stat().st_mtime, reverse=True)

    if not log_files:
        st.info("No training logs found yet. Run training first.")
    else:
        sel_log = st.selectbox("Select log", [str(p) for p in log_files])
        log_path = Path(sel_log)
        df, last_test, last_val_iou, last_test_iou = parse_log(log_path)

        # Metric cards
        best_miou  = df["mIoU"].max()  if not df.empty and "mIoU"  in df else 0
        best_map50 = df["mAP50"].max() if not df.empty and "mAP50" in df else 0
        best_dice  = df["dice"].max()  if not df.empty and "dice"  in df else 0
        test_miou  = last_test.get("mIoU", 0)

        c1,c2,c3,c4 = st.columns(4)
        for col, label, val, color in [
            (c1, "Best Val mIoU",  f"{best_miou:.4f}",  "#00FFD1"),
            (c2, "Best mAP50",     f"{best_map50:.4f}", "#FF6B35"),
            (c3, "Best Dice",      f"{best_dice:.4f}",  "#3FB950"),
            (c4, "Test mIoU",      f"{test_miou:.4f}",  "#58A6FF"),
        ]:
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{label}</div>
              <div class="value" style="color:{color}">{val}</div>
            </div>""", unsafe_allow_html=True)

        if not df.empty:
            st.markdown('<div class="section-title">Training Curves</div>', unsafe_allow_html=True)
            cols = [c for c in ["loss","mIoU","mAP50","dice","pixAcc"] if c in df]
            st.line_chart(df.set_index("epoch")[cols], height=300)

            st.markdown('<div class="section-title">Epoch Table</div>', unsafe_allow_html=True)
            st.dataframe(df.style.highlight_max(subset=[c for c in ["mIoU","mAP50","dice"] if c in df],
                         color="#1B2D1B"), use_container_width=True)

        class_iou = last_test_iou or last_val_iou
        if class_iou:
            st.markdown('<div class="section-title">Per-Class IoU</div>', unsafe_allow_html=True)
            iou_df = pd.DataFrame({"Class": CLASS_NAMES[:len(class_iou)], "IoU": class_iou}).set_index("Class")
            st.bar_chart(iou_df, height=260)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Before vs After Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="section-title">Upload Images & Run Evaluation</div>
    """, unsafe_allow_html=True)

    col_up1, col_up2 = st.columns(2)

    with col_up1:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown("#### 📷 BEFORE — Blurred Images")
        before_files = st.file_uploader(
            "Upload blurred images", type=["jpg","jpeg","png"],
            accept_multiple_files=True, key="before_upload",
            label_visibility="collapsed"
        )
        if before_files:
            st.success(f"✅ {len(before_files)} image(s) ready")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_up2:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown("#### 🌟 AFTER — Clear / High-Res Images")
        after_files = st.file_uploader(
            "Upload clear images", type=["jpg","jpeg","png"],
            accept_multiple_files=True, key="after_upload",
            label_visibility="collapsed"
        )
        if after_files:
            st.success(f"✅ {len(after_files)} image(s) ready")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Validate checkpoint
    ckpt_ok = ckpt_path.exists()
    if not ckpt_ok:
        st.warning(f"⚠️ Checkpoint not found: `{ckpt_path}` — update path in sidebar")

    run_col, _ = st.columns([1, 3])
    with run_col:
        st.markdown('<div class="run-btn">', unsafe_allow_html=True)
        run_btn = st.button("🚀  Run Evaluation", disabled=not (before_files and after_files and ckpt_ok))
        st.markdown('</div>', unsafe_allow_html=True)

    if run_btn and before_files and after_files:
        # Save uploads to temp dirs
        tmp_before = ROOT / "data" / "_upload_before"
        tmp_after  = ROOT / "data" / "_upload_after"
        save_uploads(before_files, tmp_before)
        save_uploads(after_files,  tmp_after)

        out_dir = ROOT / "outputs" / "before_after_eval"

        status_box = st.empty()
        status_box.markdown('<div class="status-box">⏳ Starting evaluation...</div>',
                            unsafe_allow_html=True)

        cmd = [
            sys.executable, str(ROOT / "scripts" / "before_after_eval.py"),
            "--config",     str(config_path),
            "--checkpoint", str(ckpt_path),
            "--before",     str(tmp_before),
            "--after",      str(tmp_after),
            "--output",     str(out_dir),
        ]

        log_lines = []
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, cwd=str(ROOT)) as proc:
            for line in proc.stdout:
                log_lines.append(line.rstrip())
                status_box.markdown(
                    f'<div class="status-box">{"<br>".join(log_lines[-12:])}</div>',
                    unsafe_allow_html=True
                )
            proc.wait()
            ok = proc.returncode == 0

        if ok:
            status_box.markdown(
                '<div class="status-box" style="color:#00FFD1">✅ Evaluation complete!</div>',
                unsafe_allow_html=True
            )
            st.components.v1.html(CONFETTI_JS, height=0)
            time.sleep(0.5)
            st.session_state["eval_results_path"] = str(out_dir / "results.json")
            st.rerun()
        else:
            status_box.markdown(
                '<div class="status-box" style="color:#F85149">❌ Evaluation failed — check logs above</div>',
                unsafe_allow_html=True
            )

    # ── Results display ────────────────────────────────────────────────────────
    results_path = Path(st.session_state.get(
        "eval_results_path",
        str(ROOT / "outputs" / "before_after_eval" / "results.json")
    ))

    if results_path.exists():
        data = json.loads(results_path.read_text())
        agg  = data["aggregate"]
        imgs = data["per_image"]
        cnames = agg.get("class_names", CLASS_NAMES)
        has_after = "after" in agg

        st.markdown('<div class="section-title">📊 Aggregate Results</div>', unsafe_allow_html=True)

        if has_after:
            metrics = [
                ("BEFORE mAP50",  agg["before"]["map50"],  None,                  "#F85149"),
                ("AFTER mAP50",   agg["after"]["map50"],   agg.get("map50_gain"), "#3FB950"),
                ("BEFORE mIoU",   agg["before"]["miou"],   None,                  "#F85149"),
                ("AFTER mIoU",    agg["after"]["miou"],    agg.get("miou_gain"),  "#3FB950"),
                ("BEFORE pixAcc", agg["before"]["pixel_acc"], None,               "#8B949E"),
                ("AFTER pixAcc",  agg["after"]["pixel_acc"],  None,               "#58A6FF"),
            ]
            cols = st.columns(6)
        else:
            metrics = [
                ("BEFORE mAP50",  agg["before"]["map50"],  None, "#F85149"),
                ("BEFORE mIoU",   agg["before"]["miou"],   None, "#F85149"),
                ("BEFORE pixAcc", agg["before"]["pixel_acc"], None, "#8B949E"),
            ]
            cols = st.columns(3)
        for col, (label, val, gain, color) in zip(cols, metrics):
            delta_html = ""
            if gain is not None:
                cls = "delta-pos" if gain >= 0 else "delta-neg"
                delta_html = f'<div class="delta {cls}">{gain:+.4f}</div>'
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{label}</div>
              <div class="value" style="color:{color}">{val:.4f}</div>
              {delta_html}
            </div>""", unsafe_allow_html=True)

        # Confetti if after mAP50 > 0.5
        if has_after and agg["after"]["map50"] >= 0.5:
            st.components.v1.html(CONFETTI_JS, height=0)

        st.markdown('<div class="section-title">📉 Per-Class IoU Comparison</div>', unsafe_allow_html=True)
        if has_after:
            iou_df = pd.DataFrame({
                "Class":  cnames,
                "Before": agg["before"]["per_class_iou"],
                "After":  agg["after"]["per_class_iou"],
            }).set_index("Class")
            st.bar_chart(iou_df, height=300)
        else:
            iou_df = pd.DataFrame({
                "Class":  cnames,
                "Before": agg["before"]["per_class_iou"],
            }).set_index("Class")
            st.bar_chart(iou_df, height=300)

        st.markdown('<div class="section-title">🖼️ Per-Image Breakdown</div>', unsafe_allow_html=True)
        for r in imgs:
            before = r.get("before", {})
            after = r.get("after")
            gain_map = r.get("map50_gain")
            if after is not None and gain_map is None:
                gain_map = after.get("map50", 0.0) - before.get("map50", 0.0)

            if after is not None:
                color = "#3FB950" if gain_map is not None and gain_map >= 0 else "#F85149"
                gain_html = ""
                if gain_map is not None:
                    gain_html = f'&nbsp;|&nbsp; Gain: <b style="color:{color}">{gain_map:+.3f}</b>'
                st.markdown(f"""
                <div class="pair-card">
                  <div class="pair-header">
                    <span class="tag tag-before">BEFORE</span> {r["before_file"]}
                    &nbsp;->&nbsp;
                    <span class="tag tag-after">AFTER</span> {r["after_file"]}
                  </div>
                  <span style="color:#8B949E;font-size:0.82rem">
                    mAP50: <b style="color:#F85149">{before.get('map50', 0.0):.3f}</b>
                    -> <b style="color:#3FB950">{after.get('map50', 0.0):.3f}</b>
                    &nbsp;|&nbsp;
                    mIoU: <b style="color:#F85149">{before.get('miou', 0.0):.3f}</b>
                    -> <b style="color:#3FB950">{after.get('miou', 0.0):.3f}</b>
                    {gain_html}
                  </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pair-card">
                  <div class="pair-header">
                    <span class="tag tag-before">BEFORE</span> {r["before_file"]}
                  </div>
                  <span style="color:#8B949E;font-size:0.82rem">
                    mAP50: <b style="color:#F85149">{before.get('map50', 0.0):.3f}</b>
                    &nbsp;|&nbsp;
                    mIoU: <b style="color:#F85149">{before.get('miou', 0.0):.3f}</b>
                  </span>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">🎨 Segmentation Panels</div>', unsafe_allow_html=True)
        panel_dir = ROOT / "outputs" / "before_after_eval" / "panels"
        panels = sorted(panel_dir.glob("*_panel.png")) if panel_dir.exists() else []
        if panels:
            for p in panels:
                st.image(str(p), caption=p.stem, use_container_width=True)
        else:
            st.info("Panels will appear here after evaluation runs.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Segmentation Viewer
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Upload any image to segment</div>', unsafe_allow_html=True)

    seg_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"], key="seg_viewer")

    if seg_file and ckpt_ok:
        import torch, torch.nn.functional as F
        from src.models.factory import build_model
        from src.data.transforms import build_eval_transforms
        from src.utils.config import load_config, resolve_paths
        from src.training.checkpointing import load_checkpoint

        img_bytes = np.frombuffer(seg_file.read(), np.uint8)
        img_bgr   = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        with st.spinner("Running segmentation..."):
            cfg    = resolve_paths(load_config(str(config_path)), ROOT)
            cfg    = set_inference_model_defaults(cfg)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if "seg_model" not in st.session_state or st.session_state.get("seg_ckpt") != str(ckpt_path):
                model = build_model(cfg).to(device)
                load_checkpoint(str(ckpt_path), model=model, device=device, prefer_ema=True)
                model.eval()
                st.session_state["seg_model"] = model
                st.session_state["seg_ckpt"]  = str(ckpt_path)

            model = st.session_state["seg_model"]
            ds_cfg = cfg.get("dataset", {})
            tfm    = build_eval_transforms(tuple(cfg["train"]["image_size"]),
                                           mean=ds_cfg.get("mean"), std=ds_cfg.get("std"))
            tensor = tfm(image=img_rgb)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                if isinstance(logits, (list, tuple)): logits = logits[-1]
                logits = F.interpolate(logits, size=img_rgb.shape[:2], mode="bilinear", align_corners=False)
                pred   = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        colors = np.array([
            [128,64,128],[107,142,35],[190,153,153],[220,20,60],
            [34,139,34],[70,130,180],[0,0,142],[244,164,96],[255,255,0],[0,191,255]
        ], dtype=np.uint8)
        seg_rgb = np.zeros((*pred.shape, 3), dtype=np.uint8)
        for i, c in enumerate(colors):
            seg_rgb[pred == i] = c

        overlay = (img_rgb * 0.45 + seg_rgb * 0.55).astype(np.uint8)

        c1, c2, c3 = st.columns(3)
        c1.image(img_rgb,  caption="Original",    use_container_width=True)
        c2.image(seg_rgb,  caption="Segmentation",use_container_width=True)
        c3.image(overlay,  caption="Overlay",     use_container_width=True)

        # Legend
        st.markdown('<div class="section-title">Class Legend</div>', unsafe_allow_html=True)
        leg_cols = st.columns(5)
        for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
            hex_c = "#{:02x}{:02x}{:02x}".format(*color)
            leg_cols[i % 5].markdown(
                f'<div style="display:flex;align-items:center;gap:6px;margin:4px 0">'
                f'<div style="width:14px;height:14px;border-radius:3px;background:{hex_c}"></div>'
                f'<span style="color:#E6EDF3;font-size:0.8rem">{name}</span></div>',
                unsafe_allow_html=True
            )
    elif seg_file and not ckpt_ok:
        st.warning("Set a valid checkpoint path in the sidebar first.")
    else:
        st.info("Upload an image above to run live segmentation.")


if __name__ == "__main__":
    pass
