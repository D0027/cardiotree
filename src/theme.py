"""Custom CSS + small UI components for the Streamlit app."""
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
from .config import C_DT, C_RF, C_ACC, C_RED, C_AMB

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, rgba(56,189,248,.10), transparent 60%),
                    radial-gradient(900px 500px at 100% 0%, rgba(167,139,250,.10), transparent 55%), #0B1220; }
.block-container { padding-top: 4rem; max-width: 1200px; }
section[data-testid="stSidebar"] { background: #0E1730; border-right: 1px solid rgba(255,255,255,.06); }
h1, h2, h3 { letter-spacing: -.02em; }
.hero { padding: 2.2rem 2.4rem; border-radius: 24px; border: 1px solid rgba(255,255,255,.08);
        background: linear-gradient(135deg, rgba(56,189,248,.14), rgba(167,139,250,.12) 60%, rgba(52,211,153,.10));
        margin-bottom: 1.4rem; }
.hero h1 { font-size: 2.8rem; font-weight: 800; margin: 0 0 .3rem 0;
           background: linear-gradient(90deg,#38BDF8,#A78BFA 55%,#34D399); -webkit-background-clip: text;
           -webkit-text-fill-color: transparent; }
.hero p { color: #AFC0DA; font-size: 1.05rem; margin: 0; max-width: 760px; }
.chip { display:inline-block; padding:.2rem .7rem; margin:.7rem .4rem 0 0; border-radius:999px; font-size:.78rem;
        font-weight:600; color:#CFE3FF; background: rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.10); }
.kpi { padding: 1.1rem 1.2rem; border-radius: 18px; background: rgba(255,255,255,.04);
       border: 1px solid rgba(255,255,255,.08); height: 100%; }
.kpi .v { font-size: 2rem; font-weight: 800; line-height: 1.1; }
.kpi .l { font-size: .78rem; color: #8FA3C2; text-transform: uppercase; letter-spacing: .08em; margin-top:.25rem; }
.kpi .s { font-size: .82rem; color: #AFC0DA; margin-top:.35rem; }
.card { padding: 1.2rem 1.4rem; border-radius: 18px; background: rgba(255,255,255,.04);
        border: 1px solid rgba(255,255,255,.08); margin-bottom: 1rem; }
.note { padding: .9rem 1.1rem; border-left: 3px solid #FBBF24; border-radius: 8px;
        background: rgba(251,191,36,.07); color:#E9D9A8; font-size:.9rem; }
.pill-dt { color:#38BDF8; font-weight:700; } .pill-rf { color:#34D399; font-weight:700; }
.gauge-label { text-align:center; font-weight:700; margin-top:-.4rem; }
.footer { text-align:center; color:#6F82A3; font-size:.8rem; margin-top:3rem; }
div[data-testid="stTabs"] button { font-weight:600; }
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)
    tmpl = go.layout.Template()
    tmpl.layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#CFDAEE"),
        colorway=[C_DT, C_RF, C_ACC, C_RED, C_AMB],
        xaxis=dict(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.07)", zerolinecolor="rgba(255,255,255,.12)"),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    pio.templates["cardiotree"] = tmpl
    pio.templates.default = "cardiotree"


def hero(title, subtitle, chips=()):
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{chips_html}</div>', unsafe_allow_html=True)


def kpi(col, value, label, sub="", color="#E6EDF7"):
    col.markdown(f'<div class="kpi"><div class="v" style="color:{color}">{value}</div>'
                 f'<div class="l">{label}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value * 100, number={"suffix": "%", "font": {"size": 38}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#6F82A3"}, "bar": {"color": color, "thickness": .3},
               "bgcolor": "rgba(255,255,255,.05)", "borderwidth": 0,
               "threshold": {"line": {"color": "#fff", "width": 2}, "thickness": .8, "value": 50}}))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0))
    return fig
