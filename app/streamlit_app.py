"""EarthChange AI analysis workspace. Launch with: streamlit run app/streamlit_app.py"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from earthchange.agent import analyze  # noqa: E402

st.set_page_config(page_title="EarthChange | Change Analysis", page_icon="🌍", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
:root{--ink:#17231d;--muted:#68766d;--line:#dce4dd;--green:#17674d;--paper:#f4f6f3;--white:#fff}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:var(--paper)}.block-container{max-width:1460px;padding:1rem 2rem 3rem}
h1,h2,h3{font-family:'DM Sans',sans-serif!important;letter-spacing:-.04em;color:var(--ink)}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:5px 0 16px;border-bottom:1px solid var(--line);margin-bottom:26px}
.brand{font:700 17px 'DM Sans',sans-serif;letter-spacing:-.055em}.brand span{color:var(--green)}
.eyebrow{font:600 11px 'IBM Plex Mono',monospace;color:var(--green);margin-bottom:10px}
.hero{padding:2px 0 23px}.hero h1{font-size:clamp(28px,3vw,40px);line-height:1.08;margin:0 0 9px}.hero p{color:var(--muted);margin:0;font-size:15px}
.panel{background:var(--white);border:1px solid var(--line);border-radius:6px;padding:18px 20px;margin-bottom:14px}
.panel-title{font-weight:700;font-size:16px;margin-bottom:4px}.subtle{color:var(--muted);font-size:13px}
.result{background:#e8f0e9;border-left:3px solid var(--green);padding:18px 22px;margin:12px 0 18px;line-height:1.6}
.result-label{font:600 11px 'IBM Plex Mono',monospace;color:var(--green);margin-bottom:7px}
div[data-testid="stMetric"]{background:white;border:1px solid var(--line);padding:15px;border-radius:6px}
div[data-testid="stMetricLabel"]{color:var(--muted)}
div.stButton>button{background:var(--green);color:white;border:1px solid var(--green);border-radius:5px;font-weight:600;min-height:44px}
div.stButton>button:hover{background:#104f3b;color:white;border-color:#104f3b}
div.stButton>button:focus-visible{outline:3px solid #87b99f;outline-offset:2px}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid var(--line)}
[data-testid="stFileUploader"]{border:1px dashed #bac9bd;border-radius:5px;background:#f9fbf9}
@media(max-width:700px){.block-container{padding:.8rem 1rem 2rem}.topbar{margin-bottom:19px}}
</style>
<div class="topbar"><div class="brand">Earth<span>Change</span> <span style="color:#68766d;font-weight:500">/ Workspace</span></div><div class="subtle">Satellite image analysis</div></div>
<div class="hero"><div class="eyebrow">Change analysis</div><h1>Read the landscape over time.</h1><p>Compare two aligned captures and inspect where the surface changed.</p></div>
""", unsafe_allow_html=True)


def load_image(file) -> np.ndarray:
    raw = file.getvalue()
    if Path(file.name).suffix.lower() in {".tif", ".tiff"}:
        import rasterio
        from rasterio.io import MemoryFile
        with MemoryFile(raw) as mem:
            with mem.open() as dataset:
                data = dataset.read()
    else:
        arr = np.asarray(Image.open(BytesIO(raw)))
        if arr.ndim == 2:
            arr = arr[:, :, None]
        data = np.moveaxis(arr, -1, 0)
    if data.ndim != 3:
        raise ValueError("Image must contain one or more bands")
    return np.asarray(data, dtype=np.float32)


def preview(data: np.ndarray):
    st.image(data[0] if data.shape[0] == 1 else np.moveaxis(data[:3], 0, -1), clamp=True, use_container_width=True)


with st.sidebar:
    st.markdown('<div class="eyebrow">INPUT DATA</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Image pair</div><div class="subtle">Use matching dimensions and aligned captures.</div>', unsafe_allow_html=True)
    before = st.file_uploader("Before · earlier capture", type=["png", "jpg", "jpeg", "tif", "tiff"], key="before")
    after = st.file_uploader("After · later capture", type=["png", "jpg", "jpeg", "tif", "tiff"], key="after")
    st.markdown('<div class="eyebrow" style="margin-top:22px">QUESTION</div>', unsafe_allow_html=True)
    question = st.text_area("Analysis question", "What percentage of the image changed?", height=90)
    pixel_size = st.number_input("Ground sampling distance (m / pixel)", min_value=0.0, value=0.0, step=0.1, help="Leave at 0 when the image resolution is unknown.")
    run = st.button("Analyze image pair", use_container_width=True, disabled=not (before and after))
    st.caption("PNG, JPEG, or GeoTIFF. Images must share dimensions and alignment.")

if not before or not after:
    st.markdown('<div class="panel"><div class="panel-title">Add a capture pair to begin</div><div class="subtle">Choose an earlier and a later image in the input panel. Matching, aligned images produce the most meaningful comparison.</div></div>', unsafe_allow_html=True)
    st.markdown("**Try asking**  \nWhere did changes occur?  ·  How much of the image changed?  ·  Compare the left and right regions.")
    st.stop()

try:
    a, b = load_image(before), load_image(after)
except Exception as exc:
    st.error(f"Unable to read image: {exc}")
    st.stop()
if a.shape != b.shape:
    st.error(f"Image dimensions or band counts differ: {a.shape} vs {b.shape}. Use a matching, aligned pair.")
    st.stop()

st.markdown('<div class="panel"><div class="panel-title">Capture pair</div><div class="subtle">Dimensions: {:,} × {:,} px &nbsp;·&nbsp; {} bands</div>'.format(a.shape[2], a.shape[1], a.shape[0]), unsafe_allow_html=True)
left, right = st.columns(2, gap="medium")
with left:
    st.caption(f"BEFORE  /  {before.name}")
    preview(a)
with right:
    st.caption(f"AFTER  /  {after.name}")
    preview(b)
st.markdown('</div>', unsafe_allow_html=True)

if run:
    out = PROJECT_ROOT / "experiments" / "outputs" / "app"
    with st.spinner("Measuring image change…"):
        result = analyze(a, b, question, pixel_resolution=(pixel_size, pixel_size) if pixel_size else None, output_dir=out)
    if not result.success:
        st.error(result.error or result.answer)
        st.stop()
    st.markdown('<div class="eyebrow">FINDINGS</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="result"><div class="result-label">Evidence-based response</div>{result.answer}</div>', unsafe_allow_html=True)
    stats = result.evidence.change_stats
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Changed area", f"{stats.changed_percentage:.2f}%")
        c2.metric("Changed pixels", f"{stats.changed_pixels:,}")
        c3.metric("Image pixels", f"{stats.total_pixels:,}")
    viz = result.evidence.visualization_paths.get("change_analysis")
    if viz and Path(viz).exists():
        st.markdown('<div class="eyebrow" style="margin-top:26px">VISUAL EVIDENCE</div>', unsafe_allow_html=True)
        st.image(viz, use_container_width=True)
    with st.expander("Analysis record"):
        st.write(f"**Intent:** {result.intent} · **Region:** {result.region} · **Operation:** {result.operation}")
        st.write("**Tools:** " + " → ".join(result.plan_steps))
        if result.evidence.threshold_used is not None:
            st.write(f"**Detection threshold:** {result.evidence.threshold_used:.4f}")
