import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optimal Husleje – Demo", layout="wide")
st.title("🏠 Optimal husleje – demo (Gradient Boosting)")

st.markdown("""
Denne demo viser, hvordan man kan:
1) træne en model til at forudsige sandsynligheden for udlejning inden 30 dage,
2) lave pris-sweep for en valgt lejlighed,
3) finde den pris, der maksimerer forventet dækningsbidrag (DB).
""")

# --- Sidebar: Data upload
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload data (CSV)", type=["csv"])
if uploaded is None:
    st.sidebar.info("Ingen fil valgt – bruger sample_data.csv")
    df = pd.read_csv("sample_data.csv")
else:
    df = pd.read_csv(uploaded)

# --- Basic checks
required_cols = [
    "unit_id","dato_start","listepris","udlejet_inden_30_dage",
    "m2","vaerelser","etage","altan","elevator","udsigt","orientering","postnr","energi_maerke","klik_pr_dag"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Mangler kolonner i CSV: {missing}")
    st.stop()

# Parse dates & sort
df["dato_start"] = pd.to_datetime(df["dato_start"])
df = df.sort_values("dato_start").reset_index(drop=True)

# Preview
with st.expander("Vis data (første 50 rækker)"):
    st.dataframe(df.head(50))

# --- Model setup
num_cols = ["m2","vaerelser","etage","klik_pr_dag","listepris"]
cat_cols = ["altan","elevator","udsigt","orientering","postnr","energi_maerke"]
target = "udlejet_inden_30_dage"

# Preprocess
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

gbm = lgb.LGBMClassifier(
    n_estimators=700,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=10,
    random_state=42
)

pipe = Pipeline([("pre", pre), ("model", gbm)])

# Time-based CV
splits = st.sidebar.slider("CV-splits (time-based)", 3, 6, 4)
tscv = TimeSeriesSplit(n_splits=splits)

auc_scores = []
X = df[num_cols + cat_cols]
y = df[target].astype(int)

for tr_idx, te_idx in tscv.split(X):
    pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
    pr = pipe.predict_proba(X.iloc[te_idx])[:,1]
    auc = roc_auc_score(y.iloc[te_idx], pr)
    auc_scores.append(auc)

col1, col2 = st.columns(2)
with col1:
    st.metric("AUC (Time CV)", f"{np.mean(auc_scores):.3f}", f"±{np.std(auc_scores):.3f}")

# Fit on all data (for demo simplicity)
pipe.fit(X, y)

# --- Price sweep
st.header("Prisoptimering for en enhed")
units = sorted(df["unit_id"].unique().tolist())
unit_sel = st.selectbox("Vælg unit_id", units)

# Find seneste observation for den valgte enhed
u_last = df[df["unit_id"]==unit_sel].iloc[-1]
base = u_last[num_cols + cat_cols].to_dict()

st.markdown("**Parametre**")
c1, c2, c3 = st.columns(3)
with c1:
    p_min = st.number_input("Pris min (kr./md.)", value=max(5000, int(df["listepris"].quantile(0.1)) ), step=250)
with c2:
    p_max = st.number_input("Pris max (kr./md.)", value=int(df["listepris"].quantile(0.9)), step=250)
with c3:
    p_step = st.number_input("Pris step", value=250, step=50, min_value=50)

tomgang_mdr = st.slider("Tomgangsomkostning (antal måneders leje, ved ikke-udlejning)", 0.0, 2.0, 0.5, 0.05)
prices = list(range(int(p_min), int(p_max)+1, int(p_step)))

def predict_prob(price):
    row = base.copy()
    row["listepris"] = price
    row["altan"] = int(row["altan"])
    row["elevator"] = int(row["elevator"])
    row["postnr"] = int(row["postnr"])
    row_df = pd.DataFrame([row])
    return pipe.predict_proba(row_df)[0,1]

res = []
for p in prices:
    pr = predict_prob(p)
    expected_db = p*pr - (tomgang_mdr*p)*(1-pr)
    res.append({"pris": p, "p_udlejet_30d": pr, "forv_DB": expected_db})

res_df = pd.DataFrame(res)
best_idx = res_df["forv_DB"].idxmax()
best_row = res_df.loc[best_idx]

c1, c2 = st.columns(2)
with c1:
    st.subheader("Sandsynlighed for udlejning ≤ 30 dage")
    fig1, ax1 = plt.subplots()
    ax1.plot(res_df["pris"], res_df["p_udlejet_30d"])
    ax1.set_xlabel("Pris (kr./md.)")
    ax1.set_ylabel("p(≤30 dage)")
    st.pyplot(fig1)

with c2:
    st.subheader("Forventet dækningsbidrag")
    fig2, ax2 = plt.subplots()
    ax2.plot(res_df["pris"], res_df["forv_DB"])
    ax2.axvline(best_row["pris"], linestyle="--")
    ax2.set_xlabel("Pris (kr./md.)")
    ax2.set_ylabel("Forv. DB (kr./md.)")
    st.pyplot(fig2)

st.success(f"**Anbefalet pris:** {int(best_row['pris'])} kr./md.  \n"
           f"p(udlejet ≤30d) ≈ {best_row['p_udlejet_30d']:.1%}  \n"
           f"Forv. DB ≈ {int(best_row['forv_DB']):,} kr./md.".replace(",", "."))

# --- Feature importance (global)
st.header("Hvad driver modellen? (Feature importance)")
enc = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
num_out = num_cols
cat_out = enc.get_feature_names_out(cat_cols).tolist()
feat_names = num_out + cat_out

booster = pipe.named_steps["model"]
imp = booster.feature_importances_
fi = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False).head(20)
st.dataframe(fi.reset_index(drop=True))

st.caption("Demo-app – ikke juridisk rådgivning. Justér parametre og brug egne data for realistiske resultater.")
