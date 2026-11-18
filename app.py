"""CardioTree — an interactive lab for Decision Trees & Random Forests."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.tree import export_graphviz, export_text

from src import data, models, theme
from src.config import (C_ACC, C_AMB, C_DT, C_RED, C_RF, CLASS_NAMES, DT_TUNED, FEATURE_INFO, RS)

st.set_page_config(page_title="CardioTree · Trees & Forests Lab", page_icon="🌳", layout="wide")
theme.inject()


# ───────────────────────── cached data / models ─────────────────────────
@st.cache_data
def get_data():
    df, n_dup = data.load_clean()
    return df, n_dup, data.split(df)


@st.cache_resource
def get_models():
    _, _, (X, y, X_tr, X_te, y_tr, y_te, feats) = get_data()
    return models.train_all(X_tr, y_tr)


@st.cache_data
def cached_depth(): _, _, (X, y, a, b, c, d, f) = get_data(); return models.depth_sweep(a, c, b, d)
@st.cache_data
def cached_ccp(): _, _, (X, y, a, b, c, d, f) = get_data(); return models.ccp_sweep(a, c, b, d)
@st.cache_data
def cached_trees(): _, _, (X, y, a, b, c, d, f) = get_data(); return models.trees_sweep(a, c, b, d)


@st.cache_data
def cached_cv():
    _, _, (X, y, *_rest) = get_data()
    return models.repeated_cv(get_models(), X, y)


@st.cache_data
def cached_perm():
    _, _, (X, y, X_tr, X_te, y_tr, y_te, feats) = get_data()
    return models.permutation_imp(get_models()["Random Forest (tuned)"], X_te, y_te, feats)


df, n_dup, (X, y, X_tr, X_te, y_tr, y_te, feats) = get_data()
M = get_models()
dt, rf = M["Decision Tree (tuned)"], M["Random Forest (tuned)"]
comp = models.compare(M, X_tr, y_tr, X_te, y_te)

# ───────────────────────── sidebar ─────────────────────────
PAGES = ["🏠 Overview", "🔎 Data Explorer", "🌳 Tree Lab", "📉 Overfitting Lab",
         "🌲 Forest vs Tree", "⭐ Feature Importance", "🧪 Try a Patient", "📚 Concepts"]
with st.sidebar:
    st.markdown("## 🌳 CardioTree")
    st.caption("Decision Trees & Random Forests, hands-on.")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    st.markdown(f"**Dataset** · {len(df)} patients · {len(feats)} features")
    st.markdown(f"**Split** · {len(X_tr)} train / {len(X_te)} test")
    st.markdown(f"**Seed** · `{RS}`")
    st.caption("Built with scikit-learn, Plotly & Streamlit.")


# ───────────────────────── pages ─────────────────────────
def page_overview():
    theme.hero("CardioTree",
               "An interactive lab that shows how a single decision tree learns, why it overfits, "
               "and how a random forest fixes it — on the classic Heart Disease dataset.",
               ["Decision Trees", "Random Forests", "Pruning", "Feature Importance", "Cross-Validation"])
    best = comp["test_acc"].idxmax()
    c = st.columns(4)
    theme.kpi(c[0], f"{len(df)}", "Patients", f"{n_dup} duplicate removed", C_DT)
    theme.kpi(c[1], f"{comp.loc['Decision Tree (unrestricted)', 'train_acc']:.0%} → "
                    f"{comp.loc['Decision Tree (unrestricted)', 'test_acc']:.0%}",
              "Unpruned tree: train → test", "the overfitting gap", C_RED)
    theme.kpi(c[2], f"{comp.loc['Random Forest (tuned)', 'test_acc']:.1%}", "Random Forest test accuracy",
              f"ROC-AUC {comp.loc['Random Forest (tuned)', 'roc_auc']:.3f}", C_RF)
    theme.kpi(c[3], f"{rf.oob_score_:.1%}", "Out-of-bag accuracy", "free validation from bagging", C_ACC)

    st.markdown("### Model scoreboard")
    long = comp.reset_index().melt("model", ["train_acc", "test_acc"], var_name="split", value_name="accuracy")
    long["split"] = long["split"].map({"train_acc": "Train", "test_acc": "Test"})
    fig = px.bar(long, x="model", y="accuracy", color="split", barmode="group",
                 color_discrete_map={"Train": C_RED, "Test": C_DT}, text_auto=".2f")
    fig.update_layout(height=380, yaxis_range=[0, 1.08], xaxis_title=None, legend_title=None)
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    a.markdown('<div class="card"><b>The story in 3 lines</b><br>'
               '1. A deep tree memorises training data (100% train, far lower test).<br>'
               '2. Pruning / depth limits trade a bit of training fit for generalisation.<br>'
               '3. A forest averages many decorrelated trees → higher, steadier accuracy.</div>',
               unsafe_allow_html=True)
    b.markdown('<div class="card"><b>Explore</b><br>Use the sidebar: build your own tree in the <b>Tree Lab</b>, '
               'watch overfitting appear in the <b>Overfitting Lab</b>, or score a made-up patient in '
               '<b>Try a Patient</b>.</div>', unsafe_allow_html=True)
    st.markdown('<div class="note">Educational project only — not medical advice. The two classes are '
                'called <b>target 0 / target 1</b>: public copies of this dataset disagree on the label '
                'direction, so no clinical claim is made.</div>', unsafe_allow_html=True)


def page_data():
    theme.hero("Data Explorer", "Know your data before modelling it.")
    t1, t2, t3, t4 = st.tabs(["Table", "Distributions", "Correlations", "Data dictionary"])
    with t1:
        st.dataframe(df, width="stretch", height=360)
        counts = y.value_counts().sort_index()
        c = st.columns(3)
        theme.kpi(c[0], len(df), "Rows (after dedup)", f"{n_dup} duplicate dropped")
        theme.kpi(c[1], f"{counts[0]} / {counts[1]}", "target 0 / target 1", f"{counts[1] / len(y):.1%} are class 1")
        theme.kpi(c[2], int(df.isnull().sum().sum()), "Missing values", "nothing to impute")
        st.caption("Why dedupe? Copies of one patient landing in both train and test let a deep tree "
                   "memorise its way to inflated accuracy (data leakage).")
    with t2:
        col = st.selectbox("Feature", feats, format_func=lambda f: f"{f} — {FEATURE_INFO[f][0]}")
        fig = px.histogram(df, x=col, color=df["target"].map({0: "target 0", 1: "target 1"}), barmode="overlay",
                           opacity=.7, nbins=25, color_discrete_sequence=[C_AMB, C_DT],
                           marginal="box")
        fig.update_layout(height=430, legend_title=None)
        st.plotly_chart(fig, width="stretch")
    with t3:
        corr = df.corr()
        fig = px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(height=620)
        st.plotly_chart(fig, width="stretch")
        s = corr["target"].drop("target").sort_values()
        st.markdown("**Correlation with target**")
        st.bar_chart(s, color=C_ACC)
    with t4:
        st.dataframe(pd.DataFrame([(k, v[0], v[1]) for k, v in FEATURE_INFO.items()],
                                  columns=["column", "name", "meaning"]), hide_index=True, width="stretch")


def page_tree_lab():
    theme.hero("Tree Lab", "Grow your own decision tree. Change the knobs and watch it (and its score) respond.")
    side, main = st.columns([1, 3])
    with side:
        crit = st.radio("Split criterion", ["gini", "entropy"], horizontal=True)
        depth = st.slider("max_depth", 1, 12, 3)
        leaf = st.slider("min_samples_leaf", 1, 30, 10)
        alpha = st.slider("ccp_alpha (pruning)", 0.0, 0.05, 0.0, 0.0025, format="%.4f")
        st.caption("Notebook's grid-search winner: entropy · depth 3 · min leaf 10 · α 0.02")
    t = models.make_tree(criterion=crit, max_depth=depth, min_samples_leaf=leaf, ccp_alpha=alpha).fit(X_tr, y_tr)
    with main:
        c = st.columns(4)
        theme.kpi(c[0], f"{t.score(X_tr, y_tr):.1%}", "Train accuracy", color=C_RED)
        theme.kpi(c[1], f"{t.score(X_te, y_te):.1%}", "Test accuracy", color=C_DT)
        theme.kpi(c[2], t.get_depth(), "Actual depth")
        theme.kpi(c[3], t.get_n_leaves(), "Leaves")
        tab1, tab2, tab3 = st.tabs(["Tree diagram", "Rules (text)", "Confusion matrix"])
        with tab1:
            if t.get_n_leaves() > 60:
                st.warning("Large tree — showing the top 4 levels only.")
                dot = export_graphviz(t, max_depth=4, feature_names=feats, class_names=CLASS_NAMES,
                                      filled=True, rounded=True, special_characters=True)
            else:
                dot = export_graphviz(t, feature_names=feats, class_names=CLASS_NAMES,
                                      filled=True, rounded=True, special_characters=True)
            st.graphviz_chart(dot, width="stretch")
            st.download_button("⬇ Download .dot file", dot, "decision_tree.dot")
        with tab2:
            st.code(export_text(t, feature_names=feats, max_depth=6), language="text")
        with tab3:
            cm = confusion_matrix(y_te, t.predict(X_te))
            fig = px.imshow(cm, text_auto=True, x=CLASS_NAMES, y=CLASS_NAMES, color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Actual"))
            fig.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")


def page_overfit():
    theme.hero("Overfitting Lab", "Train accuracy always climbs. Honest validation scores don't.")
    t1, t2 = st.tabs(["Depth sweep", "Cost-complexity pruning"])
    with t1:
        d = cached_depth()
        bd = int(d.loc[d["cv"].idxmax(), "max_depth"])
        fig = go.Figure()
        for col, name, color in [("train", "Train", C_RED), ("cv", "Cross-validation", C_DT), ("test", "Test", C_RF)]:
            fig.add_scatter(x=d["max_depth"], y=d[col], name=name, mode="lines+markers",
                            line=dict(color=color, width=3))
        fig.add_vline(x=bd, line_dash="dot", line_color="#8FA3C2", annotation_text=f"best CV depth = {bd}")
        fig.update_layout(height=450, xaxis_title="max_depth", yaxis_title="Accuracy", title="Overfitting vs tree depth")
        st.plotly_chart(fig, width="stretch")
        st.caption("Where the train curve keeps rising but CV/test flatten or fall, the tree is memorising noise.")
    with t2:
        c = cached_ccp()
        ba = float(c.loc[c["cv"].idxmax(), "ccp_alpha"])
        fig = go.Figure()
        for col, name, color in [("train", "Train", C_RED), ("cv", "Cross-validation", C_DT), ("test", "Test", C_RF)]:
            fig.add_scatter(x=c["ccp_alpha"], y=c[col], name=name, mode="lines+markers", line=dict(color=color, width=3))
        fig.add_vline(x=ba, line_dash="dot", line_color="#8FA3C2", annotation_text=f"best α = {ba:.4f}")
        fig.update_layout(height=450, xaxis_title="ccp_alpha (higher = more pruning)", yaxis_title="Accuracy",
                          title="Post-pruning with ccp_alpha")
        st.plotly_chart(fig, width="stretch")
        fig2 = px.line(c, x="ccp_alpha", y="leaves", markers=True, title="Tree size shrinks as α grows")
        fig2.update_traces(line_color=C_ACC)
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, width="stretch")


def page_forest():
    theme.hero("Forest vs Tree", "Many decorrelated trees, averaged, beat one tree on every metric that matters.")
    t1, t2, t3, t4 = st.tabs(["Metrics", "ROC curves", "How many trees?", "Repeated CV"])
    with t1:
        st.dataframe(comp.style.format("{:.3f}").background_gradient(cmap="Blues", subset=comp.columns[1:]),
                     width="stretch")
        a, b = st.columns(2)
        for col, (name, m, cs) in zip((a, b), [("Decision Tree (tuned)", dt, "Blues"), ("Random Forest (tuned)", rf, "Greens")]):
            cm = confusion_matrix(y_te, m.predict(X_te))
            fig = px.imshow(cm, text_auto=True, x=CLASS_NAMES, y=CLASS_NAMES, color_continuous_scale=cs,
                            labels=dict(x="Predicted", y="Actual"), title=name)
            fig.update_layout(height=360, coloraxis_showscale=False)
            col.plotly_chart(fig, width="stretch")
    with t2:
        fig = go.Figure()
        for (name, m), color in zip(M.items(), [C_RED, C_AMB, C_DT, C_RF]):
            p = m.predict_proba(X_te)[:, 1]
            fpr, tpr, _ = roc_curve(y_te, p)
            fig.add_scatter(x=fpr, y=tpr, name=f"{name} (AUC {roc_auc_score(y_te, p):.3f})", line=dict(color=color, width=3))
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#6F82A3"), showlegend=False)
        fig.update_layout(height=520, xaxis_title="False positive rate", yaxis_title="True positive rate", title="ROC curves")
        st.plotly_chart(fig, width="stretch")
    with t3:
        n = cached_trees()
        fig = go.Figure()
        fig.add_scatter(x=n["n_estimators"], y=n["test"], name="Test", mode="lines+markers", line=dict(color=C_DT, width=3))
        fig.add_scatter(x=n["n_estimators"], y=n["oob"], name="Out-of-bag", mode="lines+markers",
                        line=dict(color=C_RF, width=3, dash="dash"))
        fig.update_layout(height=430, xaxis_type="log", xaxis_title="Number of trees (log)", yaxis_title="Accuracy",
                          title="More trees → more stable accuracy")
        st.plotly_chart(fig, width="stretch")
    with t4:
        cv_df, scores = cached_cv()
        st.caption("Repeated stratified 5-fold CV (5 × 3 = 15 scores per model) on the full dataset.")
        fig = go.Figure()
        for (name, s), color in zip(scores.items(), [C_RED, C_AMB, C_DT, C_RF]):
            fig.add_box(y=s, name=name.replace(" (", "<br>("), marker_color=color, boxpoints="all", jitter=.4)
        fig.update_layout(height=470, showlegend=False, yaxis_title="Accuracy")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(cv_df.round(4), hide_index=True, width="stretch")
        st.markdown('<div class="note">Tuned hyper-parameters were picked on the training split, which overlaps the CV '
                    'folds, so tuned scores are slightly optimistic. Nested CV would remove that bias.</div>',
                    unsafe_allow_html=True)


def page_importance():
    theme.hero("Feature Importance", "What does the model actually lean on?")
    imp = pd.DataFrame({"Decision Tree (tuned)": dt.feature_importances_,
                        "Random Forest (tuned)": rf.feature_importances_}, index=feats)
    t1, t2, t3 = st.tabs(["Impurity-based", "Permutation (test set)", "Information gain"])
    with t1:
        a, b = st.columns(2)
        for col, name, color in zip((a, b), imp.columns, (C_DT, C_RF)):
            s = imp[name].sort_values()
            fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=color))
            fig.update_layout(height=470, title=name)
            col.plotly_chart(fig, width="stretch")
    with t2:
        p = cached_perm().sort_values("mean")
        fig = go.Figure(go.Bar(x=p["mean"], y=p.index, orientation="h", marker_color=C_ACC,
                               error_x=dict(type="data", array=p["std"], color="#8FA3C2")))
        fig.update_layout(height=500, xaxis_title="Drop in test accuracy when shuffled", title="Permutation importance (RF)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Importance ≠ causation. Correlated features share (and dilute) their importance.")
    with t3:
        g = models.best_gain_per_feature(X_tr, y_tr.values)
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=C_AMB))
        fig.update_layout(height=500, xaxis_title="Best single-split information gain (bits)",
                          title="Which feature gives the best first split?")
        st.plotly_chart(fig, width="stretch")
        st.latex(r"IG = H(\text{parent}) - \left(\tfrac{n_L}{n}H(\text{left}) + \tfrac{n_R}{n}H(\text{right})\right)")


def page_patient():
    theme.hero("Try a Patient", "Enter measurements and compare what the tree and the forest say — and why.")
    st.markdown('<div class="note">Demo for learning about models only. Not a diagnostic tool.</div>',
                unsafe_allow_html=True)
    st.write("")
    med = df.median()
    c = st.columns(4)
    v = {}
    v["age"] = c[0].slider("Age", 25, 80, int(med["age"]))
    v["sex"] = c[1].selectbox("Sex", [1, 0], format_func=lambda x: "Male (1)" if x else "Female (0)")
    v["cp"] = c[2].selectbox("Chest pain type (cp)", [0, 1, 2, 3], index=int(med["cp"]))
    v["trestbps"] = c[3].slider("Resting BP", 90, 200, int(med["trestbps"]))
    v["chol"] = c[0].slider("Cholesterol", 120, 570, int(med["chol"]))
    v["fbs"] = c[1].selectbox("Fasting sugar > 120", [0, 1], format_func=lambda x: "Yes" if x else "No")
    v["restecg"] = c[2].selectbox("Resting ECG", [0, 1, 2])
    v["thalach"] = c[3].slider("Max heart rate", 70, 210, int(med["thalach"]))
    v["exang"] = c[0].selectbox("Exercise angina", [0, 1], format_func=lambda x: "Yes" if x else "No")
    v["oldpeak"] = c[1].slider("ST depression (oldpeak)", 0.0, 6.5, float(med["oldpeak"]), 0.1)
    v["slope"] = c[2].selectbox("ST slope", [0, 1, 2], index=int(med["slope"]))
    v["ca"] = c[3].selectbox("Major vessels (ca)", [0, 1, 2, 3, 4])
    v["thal"] = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3], index=int(med["thal"]), key="thal")
    x = pd.DataFrame([v])[feats]
    p_dt, p_rf = dt.predict_proba(x)[0, 1], rf.predict_proba(x)[0, 1]
    st.write("")
    a, b = st.columns(2)
    a.plotly_chart(theme.gauge(p_dt, "Decision Tree", C_DT), width="stretch")
    a.markdown(f'<div class="gauge-label pill-dt">Decision Tree · P(target 1)</div>', unsafe_allow_html=True)
    b.plotly_chart(theme.gauge(p_rf, "Random Forest", C_RF), width="stretch")
    b.markdown(f'<div class="gauge-label pill-rf">Random Forest · P(target 1)</div>', unsafe_allow_html=True)
    with st.expander("🧭 Why did the tree say that? (decision path)", expanded=True):
        for i, s in enumerate(models.decision_path_text(dt, x, feats), 1):
            st.markdown(f"{i}. {s}")
    votes = np.array([t.predict(x.values)[0] for t in rf.estimators_])
    st.markdown(f"**Forest vote:** {int(votes.sum())} of {len(votes)} trees vote *target 1*.")
    st.progress(float(votes.mean()))


def page_concepts():
    theme.hero("Concepts", "The key ideas, in plain language.")
    qa = [
        ("How does a decision tree work?",
         "It asks yes/no questions of the form `feature <= threshold`, choosing at each node the split that makes "
         "the children purest (largest drop in Gini or entropy), and recurses until a stopping rule hits. "
         "A prediction follows the answers to a leaf: the majority class (or the mean for regression)."),
        ("What are entropy and information gain?",
         "Entropy `H = -Σ pᵢ log₂ pᵢ` is 0 for a pure node and 1 bit for a 50/50 binary node. Information gain is "
         "parent entropy minus the weighted entropy of the children. Gini `1 - Σ pᵢ²` is a cheaper alternative "
         "that usually picks similar splits."),
        ("Why do trees overfit and how do we stop it?",
         "Unrestricted trees keep splitting until every training sample is right, memorising noise. Limit "
         "`max_depth`, require `min_samples_leaf`, post-prune with `ccp_alpha`, or use an ensemble."),
        ("How does a random forest reduce variance?",
         "Each tree sees a bootstrap sample of rows and a random subset of features at every split. Their errors "
         "are partly independent, so averaging cancels noise while bias stays low."),
        ("What is out-of-bag (OOB) error?",
         "Each tree never sees ~37% of rows. Scoring each row only with trees that skipped it gives a free "
         "validation estimate without a separate hold-out set."),
        ("Which importance should I trust?",
         "Impurity importance is fast but biased toward continuous features and measured on training data. "
         "Permutation importance is measured on unseen data and is model-agnostic — prefer it when they disagree."),
        ("When might I NOT use trees?",
         "Tiny data with smooth linear relationships, extrapolation beyond the training range, or when a "
         "single compact, calibrated probability model is needed."),
    ]
    for q, a in qa:
        with st.expander(q):
            st.markdown(a)
    st.markdown("#### Bonus: trees for regression")
    st.caption("Leaves predict the mean target of their samples. See the notebook in `notebooks/` for the "
               "max-heart-rate regression experiment.")


{"🏠 Overview": page_overview, "🔎 Data Explorer": page_data, "🌳 Tree Lab": page_tree_lab,
 "📉 Overfitting Lab": page_overfit, "🌲 Forest vs Tree": page_forest, "⭐ Feature Importance": page_importance,
 "🧪 Try a Patient": page_patient, "📚 Concepts": page_concepts}[page]()

st.markdown('<div class="footer">CardioTree · built with scikit-learn, Plotly & Streamlit · educational use only</div>',
            unsafe_allow_html=True)
