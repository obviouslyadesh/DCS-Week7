from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

CSV_FILE = Path(__file__).parent / "kidney_disease.csv"

st.set_page_config(
    page_title="CKD Explorer",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        div[data-testid="stMetric"] {
            background-color: rgba(120, 130, 150, 0.10);
            border-radius: 10px;
            padding: 12px 16px;
            border-left: 5px solid #2a9d8f;
        }
        h1, h2, h3 { letter-spacing: -0.3px; }
    </style>
    """,
    unsafe_allow_html=True,
)


COLUMN_LABELS = {
    "age": "Age (years)",
    "bp": "Blood pressure (mm Hg)",
    "sg": "Specific gravity",
    "al": "Albumin (0-5)",
    "su": "Sugar (0-5)",
    "rbc": "Red blood cells (urine)",
    "pc": "Pus cell",
    "pcc": "Pus cell clumps",
    "ba": "Bacteria",
    "bgr": "Blood glucose random (mg/dL)",
    "bu": "Blood urea (mg/dL)",
    "sc": "Serum creatinine (mg/dL)",
    "sod": "Sodium (mEq/L)",
    "pot": "Potassium (mEq/L)",
    "hemo": "Hemoglobin (g/dL)",
    "pcv": "Packed cell volume",
    "wc": "White blood cell count (cells/cu mm)",
    "rc": "Red blood cell count (millions/cu mm)",
    "htn": "Hypertension",
    "dm": "Diabetes mellitus",
    "cad": "Coronary artery disease",
    "appet": "Appetite",
    "pe": "Pedal edema",
    "ane": "Anemia",
    "classification": "Diagnosis",
}

NUMERIC_COLS = [
    "age", "bp", "sg", "al", "su", "bgr", "bu", "sc",
    "sod", "pot", "hemo", "pcv", "wc", "rc",
]
CATEGORICAL_COLS = [
    "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane",
]


@st.cache_data(show_spinner="Loading data...")
def load_data() -> pd.DataFrame:
    """Read the CSV that sits next to this script."""
    return pd.read_csv(CSV_FILE)


@st.cache_data(show_spinner="Loading uploaded file...")
def load_upload(file) -> pd.DataFrame:
    """Optional: read a CSV the user uploads in the sidebar."""
    return pd.read_csv(file)


@st.cache_data(show_spinner="Cleaning data...")
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known quality problems in the UCI CKD file:
    stray tabs/spaces, '?' placeholders, numbers stored as text.
    """
    df = df.copy()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = (
            df[col].astype("string").str.strip().replace({"?": pd.NA, "": pd.NA})
        )

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_COLS + ["classification"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if "classification" in df.columns:
        df["classification"] = (
            df["classification"]
            .astype("string")
            .map({"ckd": "CKD", "notckd": "Not CKD"})
            .astype("category")
        )

    return df.drop(columns=["id"], errors="ignore")


st.sidebar.title("🩺 CKD Explorer")
st.sidebar.header("1. Data source")

uploaded = st.sidebar.file_uploader(
    "Optional: upload your own CSV (same columns)", type="csv"
)
if uploaded is not None:
    raw = load_upload(uploaded)
    st.sidebar.success("Using uploaded file")
else:
    raw = load_data()
    st.sidebar.caption(f"Using {CSV_FILE.name}")

df = clean(raw)

missing_required = {"classification", "age"} - set(df.columns)
if missing_required:
    st.error(f"Dataset is missing required columns: {sorted(missing_required)}")
    st.stop()

st.sidebar.header("2. Filters")

diag_options = sorted(df["classification"].dropna().unique())
diag_sel = st.sidebar.multiselect("Diagnosis", diag_options, default=diag_options)

age_min, age_max = int(df["age"].min()), int(df["age"].max())
age_sel = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))

flag_filters = {}
for code in ["htn", "dm", "ane"]:
    if code in df.columns:
        opts = sorted(df[code].dropna().unique())
        flag_filters[code] = st.sidebar.multiselect(
            COLUMN_LABELS[code], opts, default=opts, key=f"flt_{code}"
        )

mask = df["classification"].isin(diag_sel)
mask &= df["age"].between(*age_sel) | df["age"].isna()
for code, selected in flag_filters.items():
    mask &= df[code].isin(selected) | df[code].isna()
fdf = df[mask]

st.sidebar.header("3. Appearance")
palette_name = st.sidebar.selectbox(
    "Colour palette", ["Teal / Coral", "Colour-blind safe", "Blue / Orange"]
)
PALETTES = {
    "Teal / Coral": {"CKD": "#e76f51", "Not CKD": "#2a9d8f"},
    "Colour-blind safe": {"CKD": "#D55E00", "Not CKD": "#0072B2"},
    "Blue / Orange": {"CKD": "#ff7f0e", "Not CKD": "#1f77b4"},
}
COLORS = PALETTES[palette_name]
PLOTLY_TEMPLATE = st.sidebar.selectbox(
    "Plotly theme", ["plotly_white", "plotly", "plotly_dark", "ggplot2", "seaborn"]
)


st.title("Chronic Kidney Disease: Interactive Exploration")
st.caption(
    "Data: UCI Chronic Kidney Disease dataset (Rubini, Soundarapandian, & "
    "Eswaran, 2015). Use the sidebar to filter; every chart updates instantly."
)

if fdf.empty:
    st.warning("No records match the current filters.")
    st.stop()

ckd_share = (fdf["classification"] == "CKD").mean() * 100
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records shown", f"{len(fdf):,}", f"{len(fdf) - len(df):+,} vs. total", delta_color="off")
c2.metric("CKD prevalence", f"{ckd_share:.1f}%")
c3.metric("Median serum creatinine", f"{fdf['sc'].median():.2f} mg/dL")
c4.metric("Missing cells", f"{fdf.isna().mean().mean() * 100:.1f}%")


tab_overview, tab_dist, tab_rel, tab_cat, tab_data = st.tabs(
    ["📋 Overview", "📊 Distributions", "🔗 Relationships", "🧩 Categorical", "💾 Data"]
)

with tab_overview:
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Diagnosis balance")
        counts = fdf["classification"].value_counts().reset_index()
        counts.columns = ["Diagnosis", "Patients"]
        fig = px.pie(
            counts, names="Diagnosis", values="Patients", hole=0.5,
            color="Diagnosis", color_discrete_map=COLORS,
            template=PLOTLY_TEMPLATE,
        )
        fig.update_layout(margin=dict(t=10, b=10), legend_title_text="")
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Missing data by variable")
        miss = (fdf.isna().mean() * 100).sort_values(ascending=False).reset_index()
        miss.columns = ["code", "Missing (%)"]
        miss["Variable"] = miss["code"].map(COLUMN_LABELS)
        fig = px.bar(
            miss, x="Missing (%)", y="Variable", orientation="h",
            template=PLOTLY_TEMPLATE, color="Missing (%)",
            color_continuous_scale="Tealrose",
        )
        fig.update_layout(
            yaxis=dict(autorange="reversed"), height=520, coloraxis_showscale=False,
            margin=dict(t=10),
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Summary statistics")
    summary = fdf[NUMERIC_COLS].describe().T.rename(index=COLUMN_LABELS)
    st.dataframe(summary.round(2), width="stretch")


with tab_dist:
    st.subheader("Distribution of a clinical measure")
    d1, d2, d3 = st.columns(3)
    feature = d1.selectbox(
        "Variable", NUMERIC_COLS, index=NUMERIC_COLS.index("sc"),
        format_func=COLUMN_LABELS.get,
    )
    kind = d2.radio("Chart type", ["Histogram", "Box", "Violin (seaborn)"], horizontal=True)
    log_scale = d3.checkbox("Log10 transform (reduces skew)", value=False)

    plot_df = fdf[[feature, "classification"]].dropna().copy()
    xlabel = COLUMN_LABELS[feature]
    if log_scale:
        positive = plot_df[feature] > 0
        plot_df = plot_df[positive]
        plot_df[feature] = np.log10(plot_df[feature])
        xlabel = f"log10({xlabel})"

    if plot_df.empty:
        st.info("No data available for this variable with the current filters.")
    elif kind == "Histogram":
        fig = px.histogram(
            plot_df, x=feature, color="classification", nbins=30,
            barmode="overlay", opacity=0.7, marginal="rug",
            color_discrete_map=COLORS, template=PLOTLY_TEMPLATE,
            labels={feature: xlabel, "classification": "Diagnosis"},
        )
        st.plotly_chart(fig, width="stretch")
    elif kind == "Box":
        fig = px.box(
            plot_df, x="classification", y=feature, color="classification",
            points="outliers", color_discrete_map=COLORS, template=PLOTLY_TEMPLATE,
            labels={feature: xlabel, "classification": "Diagnosis"},
        )
        st.plotly_chart(fig, width="stretch")
    else:
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.violinplot(
            data=plot_df, x="classification", y=feature, hue="classification",
            palette=COLORS, inner="quartile", cut=0, legend=False, ax=ax,
        )
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel(xlabel)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Group comparison**")
    grp = (
        fdf.groupby("classification", observed=True)[feature]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(2)
    )
    st.dataframe(grp, width="stretch")

with tab_rel:
    st.subheader("Scatter plot explorer")
    r1, r2, r3 = st.columns(3)
    x_var = r1.selectbox("X axis", NUMERIC_COLS, index=NUMERIC_COLS.index("hemo"),
                         format_func=COLUMN_LABELS.get)
    y_var = r2.selectbox("Y axis", NUMERIC_COLS, index=NUMERIC_COLS.index("sc"),
                         format_func=COLUMN_LABELS.get)
    size_opt = r3.selectbox("Bubble size", ["None"] + NUMERIC_COLS,
                            format_func=lambda c: "None" if c == "None" else COLUMN_LABELS[c])

    cols_needed = [x_var, y_var, "classification"] + ([] if size_opt == "None" else [size_opt])
    sc_df = fdf[list(dict.fromkeys(cols_needed))].dropna()
    if size_opt != "None":
        sc_df = sc_df[sc_df[size_opt] > 0]

    if sc_df.empty:
        st.info("Not enough complete records for this combination.")
    else:
        fig = px.scatter(
            sc_df, x=x_var, y=y_var, color="classification",
            size=None if size_opt == "None" else size_opt,
            color_discrete_map=COLORS, template=PLOTLY_TEMPLATE, opacity=0.75,
            labels={x_var: COLUMN_LABELS[x_var], y_var: COLUMN_LABELS[y_var],
                    "classification": "Diagnosis"},
        )
        st.plotly_chart(fig, width="stretch")
        r = np.corrcoef(sc_df[x_var], sc_df[y_var])[0, 1]
        st.caption(f"Pearson r = {r:.2f} (n = {len(sc_df)} complete records)")

    st.subheader("Correlation heatmap")
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
    corr = fdf[NUMERIC_COLS].corr(method=method)
    corr.index = [COLUMN_LABELS[c].split(" (")[0] for c in corr.index]
    corr.columns = corr.index
    tri_mask = np.triu(np.ones_like(corr, dtype=bool)) 

    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.heatmap(
        corr, mask=tri_mask, annot=True, fmt=".2f", cmap="vlag",
        center=0, vmin=-1, vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
    )
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Pairwise-complete observations are used, so each cell may be based on "
        "a different number of records."
    )

with tab_cat:
    st.subheader("Categorical risk factors by diagnosis")
    cat_var = st.selectbox("Variable", CATEGORICAL_COLS, index=CATEGORICAL_COLS.index("htn"),
                           format_func=COLUMN_LABELS.get)
    norm = st.radio("Show as", ["Percent within category", "Counts"], horizontal=True)

    ct = (
        fdf.dropna(subset=[cat_var])
        .groupby([cat_var, "classification"], observed=True)
        .size()
        .reset_index(name="n")
    )
    if ct.empty:
        st.info("No data for this variable with the current filters.")
    else:
        if norm.startswith("Percent"):
            ct["value"] = ct["n"] / ct.groupby(cat_var, observed=True)["n"].transform("sum") * 100
            ylab = "% of patients"
        else:
            ct["value"] = ct["n"]
            ylab = "Patients"
        fig = px.bar(
            ct, x=cat_var, y="value", color="classification", barmode="stack",
            color_discrete_map=COLORS, template=PLOTLY_TEMPLATE, text_auto=".0f",
            labels={cat_var: COLUMN_LABELS[cat_var], "value": ylab,
                    "classification": "Diagnosis"},
        )
        st.plotly_chart(fig, width="stretch")

with tab_data:
    st.subheader("Filtered data")
    show_labels = st.checkbox("Use descriptive column names", value=True)
    view = fdf.rename(columns=COLUMN_LABELS) if show_labels else fdf
    st.dataframe(view, width="stretch", height=420)
    st.download_button(
        "⬇️ Download filtered data (CSV)",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="ckd_filtered.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Educational exploratory tool; not for clinical diagnosis. "
    "Built with Streamlit, pandas, NumPy, Plotly, seaborn and Matplotlib."
)