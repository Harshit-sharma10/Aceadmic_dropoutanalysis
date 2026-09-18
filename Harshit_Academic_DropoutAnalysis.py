"""
Academic Dropout Risk Dashboard
================================
Single-file Python application — backend (Flask + scikit-learn) AND
frontend (all HTML/CSS/JS rendered as Python f-strings, zero external templates).

Dataset : UCI "Predict Students' Dropout and Academic Success"
URL     : https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Run     : python Harshit_Academic_DropoutAnalysis.py
Deploy  : gunicorn Harshit_Academic_DropoutAnalysis:app
Browser : http://127.0.0.1:5000
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import json
import threading
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# FIX 1: Never call warnings.filterwarnings() at module level — it conflicts
# with Gunicorn's internal warning filters on Python 3.12+ (causes
# "TypeError: ignore() got an unexpected keyword argument 'type'").
# Scope it inside each function that needs it instead (see train_model).

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# FIX 2: Gunicorn forks workers AFTER module import, so bootstrap() must run
# inside the application context on the first request, not at import time.
# _bootstrapped guards ensure it runs exactly once per worker process.
_bootstrapped  = False
_bootstrap_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────────────────────
MODEL         = None
SCALER        = None
FEATURE_NAMES = None
DF            = None
LABEL_ENC     = None
MODEL_METRICS = {}

# ─────────────────────────────────────────────────────────────────────────────
# Lookup maps
# ─────────────────────────────────────────────────────────────────────────────
MARITAL_MAP = {1:"Single",2:"Married",3:"Widowed",4:"Divorced",
               5:"Facto Union",6:"Legally Separated"}
GENDER_MAP  = {0:"Female",1:"Male"}

RISK_PRECAUTIONS = {
    "Financial": {
        "factors": ["Debtor","Tuition_fees_up_to_date","Scholarship_holder"],
        "precautions": [
            "Establish an emergency bursary fund accessible within 48 hours.",
            "Automate tuition-payment reminders 30 days before deadlines.",
            "Expand scholarship pool, prioritising at-risk financial profiles.",
            "Introduce income-contingent repayment plans for tuition fees.",
            "Partner with banks to offer zero-interest student overdraft products.",
        ],
    },
    "Academic": {
        "factors": ["Curricular_units_1st_sem_approved","Curricular_units_2nd_sem_approved",
                    "Curricular_units_1st_sem_grade","Curricular_units_2nd_sem_grade",
                    "Admission_grade","Previous_qualification_grade"],
        "precautions": [
            "Assign academic mentors to students with < 50 % semester pass rate.",
            "Trigger mandatory counselling after a student's first failed semester.",
            "Offer free peer-tutoring and study-skills workshops every term.",
            "Flag students with 3+ consecutive failed units for early intervention.",
            "Provide flexible re-sit exams for students disrupted by financial stress.",
        ],
    },
    "Demographic": {
        "factors": ["Age_at_enrollment","International","Nationality",
                    "Displaced","Gender"],
        "precautions": [
            "Provide language and cultural-integration programmes for international students.",
            "Create support hubs for displaced and mature-age students.",
            "Tailor communication channels to reach evening/part-time cohorts.",
            "Run gender-disaggregated retention analysis annually.",
            "Offer subsidised childcare for student parents.",
        ],
    },
    "Social": {
        "factors": ["Unemployment_rate","GDP","Inflation_rate",
                    "Mothers_qualification","Fathers_qualification",
                    "Educational_special_needs"],
        "precautions": [
            "Run career-services drop-in clinics during high-unemployment periods.",
            "Create first-generation college-student mentorship programmes.",
            "Offer free mental-health counselling and disability-support services.",
            "Collaborate with local employers on part-time placement schemes.",
            "Run financial-literacy and budgeting workshops each semester.",
        ],
    },
    "Marital": {
        "factors": ["Marital_status"],
        "precautions": [
            "Provide flexible scheduling (online/hybrid) for married students.",
            "Connect married/divorced students with family-support counselling.",
            "Ensure campus childcare facilities are affordable and accessible.",
            "Create peer-support groups for married and single-parent students.",
            "Review attendance policies to accommodate family responsibilities.",
        ],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (used when real CSV is absent)
# ─────────────────────────────────────────────────────────────────────────────
def generate_synthetic_data(n: int = 4424) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    marital   = rng.choice([1,2,3,4,5,6], n, p=[.65,.15,.10,.05,.03,.02])
    gender    = rng.choice([0,1], n, p=[.35,.65])
    age       = rng.integers(17, 60, n)
    displaced = rng.choice([0,1], n, p=[.68,.32])
    debtor    = rng.choice([0,1], n, p=[.88,.12])
    fees_ok   = rng.choice([0,1], n, p=[.10,.90])
    scholar   = rng.choice([0,1], n, p=[.75,.25])
    intl      = rng.choice([0,1], n, p=[.97,.03])
    special   = rng.choice([0,1], n, p=[.97,.03])
    nationality = rng.choice(range(1,22), n)
    app_mode  = rng.choice(range(1,18), n)
    course    = rng.choice([33,171,8014,9003,9070,9085,9119,9130,9147,
                            9238,9254,9500,9556,9670,9773,9853,9991], n)
    prev_qual  = rng.choice(range(1,17), n)
    prev_grade = rng.uniform(95,190,n).round(2)
    adm_grade  = rng.uniform(95,190,n).round(2)
    unemp = rng.choice([7.6,9.4,10.8,11.1,13.9,15.5,16.2], n)
    infl  = rng.choice([-0.3,1.4,2.6,-0.8,0.6,1.7,3.7], n)
    gdp   = rng.choice([-4.06,-3.12,-0.92,0.32,1.74,3.51], n)
    cu1_en = rng.integers(0,10,n)
    cu1_ap = np.clip(rng.integers(0,cu1_en+1,n),0,cu1_en)
    cu1_gr = np.where(cu1_ap>0, rng.uniform(9,18,n).round(2), 0.0)
    cu2_en = rng.integers(0,10,n)
    cu2_ap = np.clip(rng.integers(0,cu2_en+1,n),0,cu2_en)
    cu2_gr = np.where(cu2_ap>0, rng.uniform(9,18,n).round(2), 0.0)
    risk = ((debtor*0.25) + ((1-fees_ok)*0.20) + ((age>25).astype(int)*0.10) +
            ((1-scholar)*0.05) + ((cu1_ap==0).astype(int)*0.20) +
            ((cu2_ap==0).astype(int)*0.20) + rng.uniform(0,0.15,n))
    p_do = np.clip(risk/risk.max(),0.05,0.95)
    p_gr = np.clip(1-p_do-0.1,0.05,0.90)
    p_en = 1-p_do-p_gr
    tmap = {0:"Dropout",1:"Enrolled",2:"Graduate"}
    target = [tmap[rng.choice([0,1,2], p=np.array([pd2,pe,pg])/max(pd2+pe+pg,1e-9))]
              for pd2,pe,pg in zip(p_do,p_en,p_gr)]
    return pd.DataFrame({
        "Marital_status":marital,"Application_mode":app_mode,
        "Application_order":rng.integers(1,10,n),"Course":course,
        "Daytime_evening_attendance":rng.choice([0,1],n,p=[.15,.85]),
        "Previous_qualification":prev_qual,"Previous_qualification_grade":prev_grade,
        "Nationality":nationality,"Mothers_qualification":rng.choice(range(1,7),n),
        "Fathers_qualification":rng.choice(range(1,7),n),
        "Mothers_occupation":rng.choice(range(0,10),n),
        "Fathers_occupation":rng.choice(range(0,10),n),
        "Admission_grade":adm_grade,"Displaced":displaced,
        "Educational_special_needs":special,"Debtor":debtor,
        "Tuition_fees_up_to_date":fees_ok,"Gender":gender,
        "Scholarship_holder":scholar,"Age_at_enrollment":age,
        "International":intl,
        "Curricular_units_1st_sem_credited":rng.integers(0,5,n),
        "Curricular_units_1st_sem_enrolled":cu1_en,
        "Curricular_units_1st_sem_evaluations":rng.integers(0,10,n),
        "Curricular_units_1st_sem_approved":cu1_ap,
        "Curricular_units_1st_sem_grade":cu1_gr,
        "Curricular_units_1st_sem_without_evaluations":rng.integers(0,3,n),
        "Curricular_units_2nd_sem_credited":rng.integers(0,5,n),
        "Curricular_units_2nd_sem_enrolled":cu2_en,
        "Curricular_units_2nd_sem_evaluations":rng.integers(0,10,n),
        "Curricular_units_2nd_sem_approved":cu2_ap,
        "Curricular_units_2nd_sem_grade":cu2_gr,
        "Curricular_units_2nd_sem_without_evaluations":rng.integers(0,3,n),
        "Unemployment_rate":unemp,"Inflation_rate":infl,"GDP":gdp,"Target":target,
    })

# ─────────────────────────────────────────────────────────────────────────────
# Data loading & model training
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    csv_path = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, sep=";")
    else:
        df = generate_synthetic_data()
    df.dropna(inplace=True)
    return df

def train_model(df: pd.DataFrame):
    import warnings
    # Scope warning suppression here only — safe from Gunicorn's filter chain
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    global MODEL, SCALER, FEATURE_NAMES, LABEL_ENC, MODEL_METRICS
    features = [c for c in df.columns if c != "Target"]
    FEATURE_NAMES = features
    X = df[features].copy()
    y = df["Target"].copy()
    LABEL_ENC = LabelEncoder()
    y_enc = LABEL_ENC.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.20, random_state=42, stratify=y_enc)
    SCALER = StandardScaler()
    X_tr = SCALER.fit_transform(X_train)
    X_te = SCALER.transform(X_test)
    MODEL = RandomForestClassifier(n_estimators=200, max_depth=12,
                                   random_state=42, n_jobs=-1)
    MODEL.fit(X_tr, y_train)
    y_pred = MODEL.predict(X_te)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=LABEL_ENC.classes_,
                                   output_dict=True)
    MODEL_METRICS = {
        "accuracy": round(acc*100, 2),
        "classes":  list(LABEL_ENC.classes_),
        "report":   report,
        "cm":       cm.tolist(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Plotly chart helpers  →  JSON string (consumed by Plotly.js in the page)
# ─────────────────────────────────────────────────────────────────────────────
_DARK_BG   = "#1e2532"
_DARK_FONT = "#e2e8f0"
_LAYOUT    = dict(plot_bgcolor=_DARK_BG, paper_bgcolor=_DARK_BG,
                  font=dict(color=_DARK_FONT), margin=dict(l=45,r=15,t=45,b=45),
                  height=310)
_COLORS    = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4"]

def _j(fig) -> str:
    return pio.to_json(fig)

def pie_chart(labels, values, title):
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.42,
        marker=dict(colors=_COLORS[:len(labels)]),
        textinfo="label+percent"))
    fig.update_layout(title=title, **_LAYOUT)
    return _j(fig)

def bar_chart(x, y, title, color="#3b82f6", ylab="Count"):
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=color,
                           text=y, textposition="outside"))
    fig.update_layout(title=title, yaxis_title=ylab, **_LAYOUT)
    return _j(fig)

def grouped_bar(categories, series_dict, title):
    fig = go.Figure()
    for i,(name,vals) in enumerate(series_dict.items()):
        fig.add_trace(go.Bar(name=name, x=categories, y=vals,
                             marker_color=_COLORS[i%len(_COLORS)]))
    fig.update_layout(barmode="group", title=title,
                      legend=dict(orientation="h",y=-0.22), **_LAYOUT)
    return _j(fig)

def hbar_chart(labels, values, title):
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color="#3b82f6",
        text=[f"{v:.3f}" for v in values], textposition="outside"))
    fig.update_layout(title=title,
                      **{**_LAYOUT, "margin": dict(l=195,r=30,t=45,b=35),
                         "height":420})
    return _j(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Analysis builders
# ─────────────────────────────────────────────────────────────────────────────
def _targets(df): return list(df["Target"].unique())

def build_financial(df):
    c = {}
    for col,cats in [("Debtor",["Non-Debtor","Debtor"]),
                     ("Tuition_fees_up_to_date",["Fees Late","Fees OK"]),
                     ("Scholarship_holder",["No Scholarship","Has Scholarship"])]:
        grp = df.groupby([col,"Target"]).size().unstack(fill_value=0)
        series = {t:[int(grp.get(t, pd.Series([0,0])).iloc[i]) for i in range(2)]
                  for t in _targets(df)}
        c[f"{col}_chart"] = grouped_bar(cats, series, f"{col.replace('_',' ')} vs Outcome")

    df2 = df.copy()
    df2["combo"] = df2["Debtor"].astype(str)+"_"+df2["Tuition_fees_up_to_date"].astype(str)
    lmap = {"0_0":"Non-Debtor\nFees Late","0_1":"Non-Debtor\nFees OK",
            "1_0":"Debtor\nFees Late","1_1":"Debtor\nFees OK"}
    fc = df2.groupby("combo").apply(
        lambda x: round((x["Target"]=="Dropout").mean()*100,1)).reset_index()
    fc.columns = ["combo","pct"]
    fc["combo"] = fc["combo"].map(lmap).fillna(fc["combo"])
    c["combo_chart"] = bar_chart(list(fc["combo"]),list(fc["pct"]),
                                 "Dropout % by Financial Profile","#ef4444","Dropout %")
    c["stats"] = {
        "debtor_dropout":    round((df[df["Debtor"]==1]["Target"]=="Dropout").mean()*100,1),
        "nondbtor_dropout":  round((df[df["Debtor"]==0]["Target"]=="Dropout").mean()*100,1),
        "feeslate_dropout":  round((df[df["Tuition_fees_up_to_date"]==0]["Target"]=="Dropout").mean()*100,1),
        "scholar_dropout":   round((df[df["Scholarship_holder"]==1]["Target"]=="Dropout").mean()*100,1),
        "noscholar_dropout": round((df[df["Scholarship_holder"]==0]["Target"]=="Dropout").mean()*100,1),
    }
    return c

def build_marital(df):
    c = {}
    df2 = df.copy()
    df2["m_label"] = df2["Marital_status"].map(MARITAL_MAP).fillna("Other")
    mt = df2.groupby(["m_label","Target"]).size().unstack(fill_value=0)
    cats = list(mt.index)
    series = {t:[int(mt.get(t,pd.Series(0,index=mt.index)).loc[ct]) for ct in cats]
              for t in _targets(df)}
    c["grouped"] = grouped_bar(cats, series, "Marital Status vs Academic Outcome")
    dist = df2["m_label"].value_counts()
    c["pie"]     = pie_chart(list(dist.index), list(dist.values),
                             "Distribution by Marital Status")
    do = df2.groupby("m_label").apply(
        lambda x: round((x["Target"]=="Dropout").mean()*100,1)).reset_index()
    do.columns = ["s","p"]
    c["dropout_rate"] = bar_chart(list(do["s"]),list(do["p"]),
                                  "Dropout Rate by Marital Status","#f59e0b","Dropout %")
    c["stats"] = {
        "most_common":  df2["m_label"].mode()[0],
        "highest_risk": do.loc[do["p"].idxmax(),"s"],
        "lowest_risk":  do.loc[do["p"].idxmin(),"s"],
    }
    return c

def build_demographic(df):
    c = {}
    gv = df["Gender"].map(GENDER_MAP).value_counts()
    c["gender_pie"] = pie_chart(list(gv.index),list(gv.values),"Gender Distribution")
    df2 = df.copy()
    df2["glabel"] = df2["Gender"].map(GENDER_MAP)
    gt = df2.groupby(["glabel","Target"]).size().unstack(fill_value=0)
    cats = list(gt.index)
    series = {t:[int(gt.get(t,pd.Series(0,index=gt.index)).loc[ct]) for ct in cats]
              for t in _targets(df)}
    c["gender_target"] = grouped_bar(cats,series,"Gender vs Academic Outcome")
    bins   = [17,20,23,26,30,40,70]
    labels = ["17-20","21-23","24-26","27-30","31-40","41+"]
    df2["age_g"] = pd.cut(df2["Age_at_enrollment"],bins=bins,labels=labels)
    ag = df2.groupby("age_g",observed=True).apply(
        lambda x: round((x["Target"]=="Dropout").mean()*100,1)).reset_index()
    ag.columns=["g","p"]
    c["age_dropout"] = bar_chart([str(a) for a in ag["g"]],list(ag["p"]),
                                 "Dropout Rate by Age Group","#8b5cf6","Dropout %")
    dt = df.groupby(["Displaced","Target"]).size().unstack(fill_value=0)
    series2 = {t:[int(dt.get(t,pd.Series([0,0])).iloc[i]) for i in range(2)]
               for t in _targets(df)}
    c["displaced"] = grouped_bar(["Not Displaced","Displaced"],series2,
                                 "Displaced Students vs Outcome")
    c["stats"] = {
        "pct_female":    round((df["Gender"]==0).mean()*100,1),
        "pct_male":      round((df["Gender"]==1).mean()*100,1),
        "avg_age":       round(df["Age_at_enrollment"].mean(),1),
        "pct_displaced": round(df["Displaced"].mean()*100,1),
        "pct_intl":      round(df["International"].mean()*100,1),
    }
    return c

def build_social(df):
    c = {}
    df2 = df.copy()
    df2["ub"] = pd.cut(df2["Unemployment_rate"],bins=[0,8,11,14,17,20],
                       labels=["<8%","8-11%","11-14%","14-17%",">17%"])
    ub = df2.groupby("ub",observed=True).apply(
        lambda x: round((x["Target"]=="Dropout").mean()*100,1)).reset_index()
    ub.columns=["b","p"]
    c["unemp"] = bar_chart(list(ub["b"].astype(str)),list(ub["p"]),
                           "Dropout Rate by Unemployment Rate","#06b6d4","Dropout %")
    df2["gb"] = pd.cut(df2["GDP"],bins=[-5,-2,0,2,5],
                       labels=["Deep Recession","Recession","Stagnant","Growth"])
    gb = df2.groupby("gb",observed=True).apply(
        lambda x: round((x["Target"]=="Dropout").mean()*100,1)).reset_index()
    gb.columns=["b","p"]
    c["gdp"] = bar_chart(list(gb["b"].astype(str)),list(gb["p"]),
                         "Dropout Rate by GDP Environment","#10b981","Dropout %")
    edu_map={1:"None",2:"Basic 4yr",3:"Basic 6yr",4:"Basic 9yr",5:"Secondary",6:"Higher"}
    df2["medu"] = df2["Mothers_qualification"].map(edu_map).fillna("Other")
    me = df2.groupby("medu").apply(
        lambda x: round((x["Target"]=="Dropout").mean()*100,1)).reset_index()
    me.columns=["e","p"]
    c["mom_edu"] = bar_chart(list(me["e"]),list(me["p"]),
                             "Dropout Rate by Mother's Education","#f59e0b","Dropout %")
    sn = df.groupby(["Educational_special_needs","Target"]).size().unstack(fill_value=0)
    series = {t:[int(sn.get(t,pd.Series([0,0])).iloc[i]) for i in range(2)]
              for t in _targets(df)}
    c["special"] = grouped_bar(["No Special Needs","Special Needs"],series,
                               "Special Needs vs Academic Outcome")
    c["stats"] = {
        "avg_unemp":   round(df["Unemployment_rate"].mean(),2),
        "avg_gdp":     round(df["GDP"].mean(),3),
        "avg_infl":    round(df["Inflation_rate"].mean(),2),
        "pct_special": round(df["Educational_special_needs"].mean()*100,1),
    }
    return c

def build_risk_analysis(df):
    drop_df = df[df["Target"]=="Dropout"]
    grad_df = df[df["Target"]=="Graduate"]
    scores  = {}
    for cat, info in RISK_PRECAUTIONS.items():
        s = c = 0
        for feat in info["factors"]:
            if feat in df.columns:
                dm = drop_df[feat].mean(); gm = grad_df[feat].mean()
                s += abs(dm-gm)/(abs(gm)+1e-9); c += 1
        scores[cat] = round(s/max(c,1), 4)
    sorted_risk = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    max_score   = max(v for _,v in sorted_risk) or 1
    imp         = MODEL.feature_importances_
    fi_df       = pd.DataFrame({"f":FEATURE_NAMES,"i":imp}).sort_values("i",ascending=False)
    top20       = fi_df.head(20)
    fi_chart    = hbar_chart(list(top20["f"])[::-1], list(top20["i"])[::-1],
                             "Top 20 Feature Importances (Random Forest)")
    risk_chart  = bar_chart(
        [x[0] for x in sorted_risk], [x[1] for x in sorted_risk],
        "Risk Contribution by Status Category","#ef4444","Normalised Risk Score")
    return {
        "scores":      scores,
        "sorted_risk": sorted_risk,
        "top_risk":    sorted_risk[0][0],
        "max_score":   max_score,
        "fi_chart":    fi_chart,
        "risk_chart":  risk_chart,
        "model_metrics": MODEL_METRICS,
    }

def predict_single(form_data: dict) -> dict:
    row = [float(form_data.get(f, 0) or 0) for f in FEATURE_NAMES]
    X   = SCALER.transform([row])
    proba     = MODEL.predict_proba(X)[0]
    pred_idx  = int(np.argmax(proba))
    pred_lbl  = LABEL_ENC.classes_[pred_idx]
    drop_idx  = list(LABEL_ENC.classes_).index("Dropout") if "Dropout" in LABEL_ENC.classes_ else 0
    drop_prob = round(float(proba[drop_idx])*100, 1)
    risk_lvl  = "High" if drop_prob>60 else ("Medium" if drop_prob>35 else "Low")
    fin_risk  = (float(form_data.get("Debtor",0))==1 or
                 float(form_data.get("Tuition_fees_up_to_date",1))==0)
    acad_risk = (float(form_data.get("Curricular_units_1st_sem_approved",5))<2 or
                 float(form_data.get("Curricular_units_2nd_sem_approved",5))<2)
    top_cat   = "Financial" if fin_risk else ("Academic" if acad_risk else "Social")
    proba_chart = bar_chart(list(LABEL_ENC.classes_),
                            [round(p*100,1) for p in proba],
                            "Prediction Probability (%)","#3b82f6","%")
    return {"prediction":pred_lbl,"dropout_prob":drop_prob,"risk_level":risk_lvl,
            "top_category":top_cat,"precautions":RISK_PRECAUTIONS[top_cat]["precautions"],
            "proba_chart":proba_chart}

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████  PYTHON-GENERATED HTML / CSS / JS  ██████████████████████
# ─────────────────────────────────────────────────────────────────────────────

# ── shared CSS (returned as a Python string) ─────────────────────────────────
def _css() -> str:
    return """
<style>
:root{--bg:#0f1623;--sf:#1e2532;--sf2:#252e3f;--bd:#2d3748;--tx:#e2e8f0;
      --mu:#94a3b8;--bl:#3b82f6;--gr:#10b981;--yw:#f59e0b;--rd:#ef4444;
      --pu:#8b5cf6;--cy:#06b6d4;}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--tx);font-family:'Segoe UI',system-ui,sans-serif;
     font-size:14px;min-height:100vh;display:flex}
.sb{width:225px;background:var(--sf);border-right:1px solid var(--bd);
    display:flex;flex-direction:column;position:fixed;top:0;bottom:0;left:0;z-index:100}
.sb-logo{padding:18px 16px;border-bottom:1px solid var(--bd)}
.sb-logo h2{font-size:15px;font-weight:700;color:var(--bl);line-height:1.4}
.sb-logo span{font-size:11px;color:var(--mu)}
.sb nav{padding:10px 0;flex:1;overflow-y:auto}
.nav-sec{padding:8px 16px 4px;font-size:10px;text-transform:uppercase;
         letter-spacing:.08em;color:var(--mu)}
.nav-a{display:flex;align-items:center;gap:9px;padding:9px 16px;color:var(--mu);
       text-decoration:none;font-size:13px;border-left:3px solid transparent;
       transition:all .15s}
.nav-a:hover,.nav-a.on{color:var(--tx);background:var(--sf2);border-left-color:var(--bl)}
.nav-a .ic{font-size:15px;width:20px;text-align:center}
.main{margin-left:225px;flex:1;display:flex;flex-direction:column;min-height:100vh}
.topbar{background:var(--sf);border-bottom:1px solid var(--bd);padding:13px 26px;
        display:flex;align-items:center;justify-content:space-between}
.topbar h1{font-size:17px;font-weight:600}
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:600}
.b-bl{background:rgba(59,130,246,.15);color:var(--bl)}
.b-gr{background:rgba(16,185,129,.15);color:var(--gr)}
.b-rd{background:rgba(239,68,68,.15);color:var(--rd)}
.b-yw{background:rgba(245,158,11,.15);color:var(--yw)}
.b-pu{background:rgba(139,92,246,.15);color:var(--pu)}
.content{padding:22px 26px;flex:1}
.sg{display:grid;grid-template-columns:repeat(auto-fill,minmax(165px,1fr));
    gap:14px;margin-bottom:22px}
.sc{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:16px 18px}
.sc .lb{font-size:10px;color:var(--mu);text-transform:uppercase;letter-spacing:.06em}
.sc .vl{font-size:26px;font-weight:700;margin-top:4px}
.sc .sb2{font-size:11px;color:var(--mu);margin-top:2px}
.cg{display:grid;grid-template-columns:repeat(auto-fill,minmax(455px,1fr));
    gap:18px;margin-bottom:22px}
.cc{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:14px 18px}
.cc h3{font-size:11px;font-weight:600;color:var(--mu);margin-bottom:10px;
       text-transform:uppercase;letter-spacing:.05em}
.cf{background:var(--sf);border:1px solid var(--bd);border-radius:10px;
    padding:14px 18px;margin-bottom:18px}
.cf h3{font-size:11px;font-weight:600;color:var(--mu);margin-bottom:10px;
       text-transform:uppercase;letter-spacing:.05em}
.ps{background:var(--sf);border:1px solid var(--bd);border-radius:10px;
    padding:18px 22px;margin-bottom:18px}
.ps h3{font-size:14px;font-weight:600;margin-bottom:12px}
.pi{display:flex;gap:11px;padding:9px 0;border-bottom:1px solid var(--bd)}
.pi:last-child{border-bottom:none}
.pn{background:var(--bl);color:#fff;border-radius:50%;width:24px;height:24px;
    display:flex;align-items:center;justify-content:center;font-size:10px;
    font-weight:700;flex-shrink:0}
.rg{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));
    gap:14px;margin-bottom:22px}
.rc{background:var(--sf);border:1px solid var(--bd);border-radius:10px;padding:16px}
.rc .ct{font-size:10px;color:var(--mu);text-transform:uppercase;letter-spacing:.06em}
.rc .sc2{font-size:24px;font-weight:700;margin:5px 0 4px}
.rb{height:6px;background:var(--sf2);border-radius:3px;overflow:hidden;margin-top:5px}
.rbf{height:100%;background:var(--rd);border-radius:3px}
.fg{display:grid;grid-template-columns:repeat(auto-fill,minmax(255px,1fr));gap:14px}
.fgrp{display:flex;flex-direction:column}
.fgrp label{font-size:11px;color:var(--mu);margin-bottom:5px;
             text-transform:uppercase;letter-spacing:.04em}
.fgrp input,.fgrp select{width:100%;background:var(--sf2);border:1px solid var(--bd);
  color:var(--tx);border-radius:7px;padding:8px 11px;font-size:13px;outline:none}
.fgrp input:focus,.fgrp select:focus{border-color:var(--bl)}
.btn{background:var(--bl);color:#fff;border:none;padding:11px 28px;border-radius:8px;
     font-size:14px;font-weight:600;cursor:pointer;margin-top:14px}
.btn:hover{background:#2563eb}
.res{background:var(--sf);border:1px solid var(--bd);border-radius:10px;
     padding:22px;margin-bottom:18px}
.res .hl{font-size:21px;font-weight:700;margin-bottom:5px}
.ig{display:grid;grid-template-columns:repeat(auto-fill,minmax(195px,1fr));
    gap:11px;margin:14px 0}
.ii{background:var(--sf2);border-radius:8px;padding:12px 15px}
.ii .k{font-size:10px;color:var(--mu);text-transform:uppercase;letter-spacing:.04em}
.ii .v{font-size:19px;font-weight:700;margin-top:3px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{background:var(--sf2);color:var(--mu);padding:9px 12px;text-align:left;
   font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:.05em}
td{padding:9px 12px;border-bottom:1px solid var(--bd);color:var(--tx)}
tr:last-child td{border-bottom:none}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:var(--sf)}
::-webkit-scrollbar-thumb{background:var(--bd);border-radius:3px}
</style>"""

# ── Plotly JS + renderChart helper (single snippet) ──────────────────────────
def _plotly_js() -> str:
    return """
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
function rc(id,js){
  try{var s=JSON.parse(js);Plotly.newPlot(id,s.data,s.layout,
  {responsive:true,displayModeBar:false});}catch(e){console.error(id,e);}
}
</script>"""

# ── sidebar nav (Python f-string, active page highlighted) ───────────────────
def _sidebar(active: str) -> str:
    def lnk(href, icon, label, key):
        cls = "nav-a on" if active == key else "nav-a"
        return f'<a href="{href}" class="{cls}"><span class="ic">{icon}</span>{label}</a>'
    return f"""
<aside class="sb">
  <div class="sb-logo">
    <h2>&#127891; Dropout<br>Risk Dashboard</h2>
    <span>UCI Academic Dataset</span>
  </div>
  <nav>
    <div class="nav-sec">Overview</div>
    {lnk('/','&#127968;','Home','home')}
    <div class="nav-sec">Status Analysis</div>
    {lnk('/financial','&#128176;','Financial Status','financial')}
    {lnk('/marital','&#128141;','Marital Status','marital')}
    {lnk('/demographic','&#127758;','Demographic Status','demographic')}
    {lnk('/social','&#129309;','Social Status','social')}
    <div class="nav-sec">Risk Engine</div>
    {lnk('/risk','&#9888;','Risk Analysis','risk')}
    {lnk('/predict','&#128302;','Predict Student','predict')}
  </nav>
</aside>"""

# ── page shell wrapper ────────────────────────────────────────────────────────
def _page(title: str, active: str, badge: str, body: str, scripts: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>{title} — Dropout Risk Dashboard</title>
{_css()}
{_plotly_js()}
</head>
<body>
{_sidebar(active)}
<div class="main">
  <div class="topbar">
    <h1>{title}</h1>
    <span class="badge b-bl">{badge}</span>
  </div>
  <div class="content">
    {body}
  </div>
</div>
{scripts}
</body>
</html>"""

# ── stat card helper ──────────────────────────────────────────────────────────
def _stat(label: str, value, sub: str = "", color: str = "var(--bl)") -> str:
    return f"""<div class="sc">
  <div class="lb">{label}</div>
  <div class="vl" style="color:{color}">{value}</div>
  <div class="sb2">{sub}</div>
</div>"""

# ── chart card helper ─────────────────────────────────────────────────────────
def _chart_card(div_id: str, json_data: str, title: str = "", full: bool = False) -> str:
    cls   = "cf" if full else "cc"
    h3    = f"<h3>{title}</h3>" if title else ""
    jdata = json_data.replace("\\", "\\\\").replace("`", "\\`")
    return f"""<div class="{cls}">
  {h3}
  <div id="{div_id}"></div>
  <script>rc('{div_id}',`{jdata}`);</script>
</div>"""

# ── precaution list ───────────────────────────────────────────────────────────
def _precautions(category: str, items: list) -> str:
    rows = "".join(
        f'<div class="pi"><div class="pn">{i+1}</div><div>{p}</div></div>'
        for i,p in enumerate(items))
    return f"""<div class="ps">
  <h3>&#9989; Precautions — {category} Risk</h3>
  {rows}
</div>"""

# ─────────────────────────────────────────────────────────────────────────────
# ██████████████████████████  PAGE RENDERERS  ██████████████████████████████
# ─────────────────────────────────────────────────────────────────────────────

def render_home(predict_result: dict = None) -> str:
    # ── overview numbers ──────────────────────────────────────────
    total    = len(DF)
    dropouts = int((DF["Target"]=="Dropout").sum())
    grads    = int((DF["Target"]=="Graduate").sum())
    enrolled = int((DF["Target"]=="Enrolled").sum())
    do_pct   = round(dropouts/total*100,1)
    acc      = MODEL_METRICS.get("accuracy",0)

    # ── OVERVIEW tab ──────────────────────────────────────────────
    pie_json = pie_chart(["Dropout","Graduate","Enrolled"],
                         [dropouts,grads,enrolled],
                         "Overall Academic Outcome Distribution")
    cm  = MODEL_METRICS.get("cm",[])
    cls = MODEL_METRICS.get("classes",[])
    cm_html = ""
    if cm and cls:
        series  = {cls[i]: [cm[i][j] for j in range(len(cls))] for i in range(len(cls))}
        cm_html = _chart_card("cm_chart", grouped_bar(cls, series, "Confusion Matrix"),
                              "Model Confusion Matrix")
    imp   = MODEL.feature_importances_
    fi_df = pd.DataFrame({"f":FEATURE_NAMES,"i":imp}).sort_values("i",ascending=False).head(10)
    fi_html = _chart_card("fi_home",
                          hbar_chart(list(fi_df["f"])[::-1], list(fi_df["i"])[::-1],
                                     "Top 10 Feature Importances"),
                          "Feature Importances")
    rpt  = MODEL_METRICS.get("report",{})
    clf_rows = ""
    for cls_name in MODEL_METRICS.get("classes",[]):
        m = rpt.get(cls_name,{})
        clf_rows += (f"<tr><td>{cls_name}</td>"
                     f"<td>{round(m.get('precision',0)*100,1)}%</td>"
                     f"<td>{round(m.get('recall',0)*100,1)}%</td>"
                     f"<td>{round(m.get('f1-score',0)*100,1)}%</td>"
                     f"<td>{int(m.get('support',0))}</td></tr>")
    ov_stats = (
        _stat("Total Students", f"{total:,}",                           "in dataset") +
        _stat("Dropouts",       f"{dropouts:,}", f"{do_pct}% of total", "var(--rd)") +
        _stat("Graduates",      f"{grads:,}",    f"{round(grads/total*100,1)}%", "var(--gr)") +
        _stat("Enrolled",       f"{enrolled:,}", f"{round(enrolled/total*100,1)}%", "var(--yw)") +
        _stat("Model Accuracy", f"{acc}%",       "Random Forest 200 trees", "var(--bl)") +
        _stat("Features Used",  str(len(FEATURE_NAMES)), "input dimensions")
    )
    tab_overview = f"""
<div class="sg">{ov_stats}</div>
<div class="cg">
  {_chart_card("home_pie", pie_json, "Outcome Distribution")}
  {cm_html}
</div>
<div class="cg">
  {fi_html}
  <div class="cc"><h3>Classification Report</h3>
    <table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th>
    <th>F1-Score</th><th>Support</th></tr></thead>
    <tbody>{clf_rows}</tbody></table>
  </div>
</div>"""

    # ── FINANCIAL tab ─────────────────────────────────────────────
    fc  = build_financial(DF)
    fst = fc["stats"]
    fin_stats = (
        _stat("Debtor Dropout %",     f"{fst['debtor_dropout']}%",   "students with debt",    "var(--rd)") +
        _stat("Non-Debtor Dropout %", f"{fst['nondbtor_dropout']}%", "no debt",               "var(--gr)") +
        _stat("Fees Late Dropout %",  f"{fst['feeslate_dropout']}%", "tuition overdue",       "var(--rd)") +
        _stat("Scholar Dropout %",    f"{fst['scholar_dropout']}%",  "has scholarship",       "var(--yw)") +
        _stat("No Scholar Dropout %", f"{fst['noscholar_dropout']}%","no scholarship",        "var(--mu)")
    )
    tab_financial = f"""
<div class="sg">{fin_stats}</div>
<div class="cg">
  {_chart_card("fin1", fc["Debtor_chart"],                    "Debtor vs Outcome")}
  {_chart_card("fin2", fc["Tuition_fees_up_to_date_chart"],   "Tuition Fees vs Outcome")}
</div>
<div class="cg">
  {_chart_card("fin3", fc["Scholarship_holder_chart"],        "Scholarship vs Outcome")}
  {_chart_card("fin4", fc["combo_chart"],                     "Dropout % by Financial Profile")}
</div>
<h3 style="margin:16px 0 8px;font-size:13px">Financial Risk Precautions</h3>
{_precautions("Financial", RISK_PRECAUTIONS["Financial"]["precautions"])}"""

    # ── MARITAL tab ───────────────────────────────────────────────
    mc  = build_marital(DF)
    mst = mc["stats"]
    mar_stats = (
        _stat("Most Common Status",  mst["most_common"],  "marital group", "var(--bl)") +
        _stat("Highest Risk Group",  mst["highest_risk"], "dropout rate",  "var(--rd)") +
        _stat("Lowest Risk Group",   mst["lowest_risk"],  "dropout rate",  "var(--gr)") +
        _stat("Total Students",      f"{total:,}",        "in dataset")
    )
    tab_marital = f"""
<div class="sg">{mar_stats}</div>
<div class="cg">
  {_chart_card("mar1", mc["grouped"],      "Marital Status vs Academic Outcome")}
  {_chart_card("mar2", mc["pie"],          "Distribution by Marital Status")}
</div>
<div class="cf">{_chart_card("mar3", mc["dropout_rate"], "Dropout Rate by Marital Status", full=True)}</div>
<h3 style="margin:16px 0 8px;font-size:13px">Marital Risk Precautions</h3>
{_precautions("Marital", RISK_PRECAUTIONS["Marital"]["precautions"])}"""

    # ── DEMOGRAPHIC tab ───────────────────────────────────────────
    dc  = build_demographic(DF)
    dst = dc["stats"]
    dem_stats = (
        _stat("Female %",        f"{dst['pct_female']}%",   "of students",  "var(--pu)") +
        _stat("Male %",          f"{dst['pct_male']}%",     "of students",  "var(--bl)") +
        _stat("Average Age",     str(dst["avg_age"]),        "at enrollment","var(--yw)") +
        _stat("Displaced %",     f"{dst['pct_displaced']}%","displaced",    "var(--rd)") +
        _stat("International %", f"{dst['pct_intl']}%",     "international","var(--cy)")
    )
    tab_demographic = f"""
<div class="sg">{dem_stats}</div>
<div class="cg">
  {_chart_card("dem1", dc["gender_pie"],    "Gender Distribution")}
  {_chart_card("dem2", dc["gender_target"], "Gender vs Academic Outcome")}
</div>
<div class="cg">
  {_chart_card("dem3", dc["age_dropout"],   "Dropout Rate by Age Group")}
  {_chart_card("dem4", dc["displaced"],     "Displaced Students vs Outcome")}
</div>
<h3 style="margin:16px 0 8px;font-size:13px">Demographic Risk Precautions</h3>
{_precautions("Demographic", RISK_PRECAUTIONS["Demographic"]["precautions"])}"""

    # ── SOCIAL tab ────────────────────────────────────────────────
    sc  = build_social(DF)
    sst = sc["stats"]
    soc_stats = (
        _stat("Avg Unemployment", f"{sst['avg_unemp']}%", "macro indicator", "var(--rd)") +
        _stat("Avg GDP Growth",   f"{sst['avg_gdp']}",    "macro indicator", "var(--gr)") +
        _stat("Avg Inflation",    f"{sst['avg_infl']}%",  "macro indicator", "var(--yw)") +
        _stat("Special Needs %",  f"{sst['pct_special']}%","of students",    "var(--pu)")
    )
    tab_social = f"""
<div class="sg">{soc_stats}</div>
<div class="cg">
  {_chart_card("soc1", sc["unemp"],   "Dropout Rate by Unemployment Rate")}
  {_chart_card("soc2", sc["gdp"],     "Dropout Rate by GDP Environment")}
</div>
<div class="cg">
  {_chart_card("soc3", sc["mom_edu"], "Dropout Rate by Mother's Education")}
  {_chart_card("soc4", sc["special"], "Special Needs vs Academic Outcome")}
</div>
<h3 style="margin:16px 0 8px;font-size:13px">Social Risk Precautions</h3>
{_precautions("Social", RISK_PRECAUTIONS["Social"]["precautions"])}"""

    # ── RISK ANALYSIS tab ─────────────────────────────────────────
    ra      = build_risk_analysis(DF)
    max_sc  = ra["max_score"]
    medals  = ["&#129351;","&#129352;","&#129353;","4th","5th"]
    rcols   = ["var(--rd)","var(--yw)","var(--gr)","var(--bl)","var(--pu)"]
    rank_cards = ""
    for idx,(cat,score) in enumerate(ra["sorted_risk"]):
        pct = int(score / max_sc * 100)
        rank_cards += (f'<div class="rc"><div class="ct">{medals[idx]} Rank #{idx+1}</div>'
                       f'<div class="sc2" style="color:{rcols[idx]}">{cat}</div>'
                       f'<div style="font-size:11px;color:var(--mu)">Score: {score:.4f}</div>'
                       f'<div class="rb"><div class="rbf" style="width:{pct}%"></div></div></div>')
    prec_all = "".join(_precautions(cat, RISK_PRECAUTIONS[cat]["precautions"])
                       for cat,_ in ra["sorted_risk"])
    mm = ra["model_metrics"]
    tab_risk = f"""
<div class="rg">{rank_cards}</div>
<div class="cg">
  {_chart_card("risk1", ra["risk_chart"], "Risk Contribution by Category")}
  {_chart_card("fi1",   ra["fi_chart"],   "Top 20 Feature Importances")}
</div>
<h3 style="margin:16px 0 8px;font-size:13px">
  &#9888; Top Risk Category: <span style="color:var(--rd)">{ra['top_risk']}</span>
  &nbsp;&mdash; Precautions by Priority
</h3>
{prec_all}
<div class="cc" style="margin-top:16px"><h3>Model Performance Summary</h3>
  <table><thead><tr><th>Metric</th><th>Value</th></tr></thead>
  <tbody>
    <tr><td>Overall Accuracy</td><td><strong>{mm.get('accuracy',0)}%</strong></td></tr>
    <tr><td>Algorithm</td><td>Random Forest (200 trees, max_depth=12)</td></tr>
    <tr><td>Training Split</td><td>80 / 20 stratified</td></tr>
    <tr><td>Features</td><td>{len(FEATURE_NAMES)}</td></tr>
  </tbody></table>
</div>"""

    # ── PREDICT tab ───────────────────────────────────────────────
    priority_feats = [
        ("Debtor",                             "Debtor (0=No, 1=Yes)",                   "0",   "0", "1"),
        ("Tuition_fees_up_to_date",            "Tuition Fees Up-to-Date (0=No, 1=Yes)",  "1",   "0", "1"),
        ("Scholarship_holder",                 "Scholarship Holder (0=No, 1=Yes)",       "0",   "0", "1"),
        ("Age_at_enrollment",                  "Age at Enrollment",                      "20",  "15","70"),
        ("Gender",                             "Gender (0=Female, 1=Male)",              "0",   "0", "1"),
        ("Displaced",                          "Displaced (0=No, 1=Yes)",                "0",   "0", "1"),
        ("International",                      "International (0=No, 1=Yes)",            "0",   "0", "1"),
        ("Educational_special_needs",          "Special Needs (0=No, 1=Yes)",            "0",   "0", "1"),
        ("Admission_grade",                    "Admission Grade",                        "130", "0","200"),
        ("Previous_qualification_grade",       "Previous Qualification Grade",           "130", "0","200"),
        ("Curricular_units_1st_sem_enrolled",  "1st Sem Units Enrolled",                 "5",   "0","10"),
        ("Curricular_units_1st_sem_approved",  "1st Sem Units Approved",                 "4",   "0","10"),
        ("Curricular_units_1st_sem_grade",     "1st Sem Average Grade",                  "13",  "0","20"),
        ("Curricular_units_2nd_sem_enrolled",  "2nd Sem Units Enrolled",                 "5",   "0","10"),
        ("Curricular_units_2nd_sem_approved",  "2nd Sem Units Approved",                 "4",   "0","10"),
        ("Curricular_units_2nd_sem_grade",     "2nd Sem Average Grade",                  "13",  "0","20"),
        ("Unemployment_rate",                  "Unemployment Rate (%)",                  "11",  "0","25"),
        ("Inflation_rate",                     "Inflation Rate (%)",                     "1.5","-5","10"),
        ("GDP",                                "GDP Growth Rate",                        "0.5", "-5","5"),
        ("Marital_status",                     "Marital Status (1=Single \u20266=Sep.)", "1",   "1", "6"),
    ]
    shown = {f[0] for f in priority_feats}
    hidden = "".join(f'<input type="hidden" name="{feat}" value="0"/>'
                     for feat in FEATURE_NAMES if feat not in shown)
    form_inputs = "".join(
        f'<div class="fgrp"><label>{lbl}</label>'
        f'<input type="number" name="{nm}" value="{dfl}" min="{mn}" max="{mx}" step="any"/></div>'
        for nm,lbl,dfl,mn,mx in priority_feats
    )
    # Inline prediction result (if POST was submitted)
    pred_result_html = ""
    if predict_result:
        rc_color = {"High":"var(--rd)","Medium":"var(--yw)","Low":"var(--gr)"}
        col      = rc_color.get(predict_result["risk_level"],"var(--bl)")
        prec_rows = "".join(
            f'<div class="pi"><div class="pn">{i+1}</div><div>{p}</div></div>'
            for i,p in enumerate(predict_result["precautions"]))
        jdata = predict_result["proba_chart"].replace("\\","\\\\").replace("`","\\`")
        pred_result_html = f"""
<div class="res" style="margin-bottom:20px">
  <div class="hl">Prediction: <span style="color:{col}">{predict_result['prediction']}</span></div>
  <div style="color:var(--mu);font-size:13px">Dropout Probability: {predict_result['dropout_prob']}%</div>
  <div class="ig">
    <div class="ii"><div class="k">Risk Level</div>
      <div class="v" style="color:{col}">{predict_result['risk_level']}</div></div>
    <div class="ii"><div class="k">Dropout Probability</div>
      <div class="v">{predict_result['dropout_prob']}%</div></div>
    <div class="ii"><div class="k">Top Risk Category</div>
      <div class="v" style="color:var(--yw)">{predict_result['top_category']}</div></div>
    <div class="ii"><div class="k">Prediction</div>
      <div class="v">{predict_result['prediction']}</div></div>
  </div>
  <div id="proba_chart"></div>
  <script>rc('proba_chart',`{jdata}`);</script>
</div>
<div class="ps" style="margin-bottom:20px">
  <h3>&#9989; Recommended Precautions ({predict_result['top_category']} Risk)</h3>
  {prec_rows}
</div>"""

    tab_predict = f"""
{pred_result_html}
<div class="cf">
  <h3>Enter Student Details to Predict Dropout Risk</h3>
  <form method="POST" action="/?tab=predict">
    {hidden}
    <div class="fg">{form_inputs}</div>
    <button type="submit" class="btn">&#128302; Predict Dropout Risk</button>
  </form>
</div>"""

    # ── tab CSS + JS (all in Python string) ───────────────────────
    tab_css_js = """
<style>
.tabs{display:flex;gap:0;border-bottom:2px solid var(--bd);margin-bottom:22px;
      flex-wrap:wrap;background:var(--sf);border-radius:10px 10px 0 0;padding:0 8px}
.tab-btn{padding:11px 18px;background:none;border:none;color:var(--mu);font-size:13px;
         font-weight:600;cursor:pointer;border-bottom:3px solid transparent;
         margin-bottom:-2px;transition:all .15s;white-space:nowrap}
.tab-btn:hover{color:var(--tx);background:var(--sf2);border-radius:6px 6px 0 0}
.tab-btn.active{color:var(--bl);border-bottom-color:var(--bl)}
.tab-panel{display:none}.tab-panel.active{display:block}
</style>
<script>
function showTab(id){
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p=>p.classList.remove('active'));
  var btn=document.querySelector('.tab-btn[data-tab="'+id+'"]');
  var panel=document.getElementById('tab-'+id);
  if(btn) btn.classList.add('active');
  if(panel) panel.classList.add('active');
  history.replaceState(null,'','/?tab='+id);
}
window.addEventListener('DOMContentLoaded',function(){
  var tab=new URLSearchParams(window.location.search).get('tab')||'overview';
  showTab(tab);
});
</script>"""

    # active tab (from POST redirect or default)
    active_tab = "predict" if predict_result else "overview"

    body = f"""
{tab_css_js}
<div class="tabs">
  <button class="tab-btn" data-tab="overview"    onclick="showTab('overview')"   >&#127968; Overview</button>
  <button class="tab-btn" data-tab="financial"   onclick="showTab('financial')"  >&#128176; Financial</button>
  <button class="tab-btn" data-tab="marital"     onclick="showTab('marital')"    >&#128141; Marital</button>
  <button class="tab-btn" data-tab="demographic" onclick="showTab('demographic')">&#127758; Demographic</button>
  <button class="tab-btn" data-tab="social"      onclick="showTab('social')"     >&#129309; Social</button>
  <button class="tab-btn" data-tab="risk"        onclick="showTab('risk')"       >&#9888;&#65039; Risk Analysis</button>
  <button class="tab-btn" data-tab="predict"     onclick="showTab('predict')"    >&#128302; Predict Student</button>
</div>
<div id="tab-overview"    class="tab-panel">{tab_overview}</div>
<div id="tab-financial"   class="tab-panel">{tab_financial}</div>
<div id="tab-marital"     class="tab-panel">{tab_marital}</div>
<div id="tab-demographic" class="tab-panel">{tab_demographic}</div>
<div id="tab-social"      class="tab-panel">{tab_social}</div>
<div id="tab-risk"        class="tab-panel">{tab_risk}</div>
<div id="tab-predict"     class="tab-panel">{tab_predict}</div>
"""
    return _page("Academic Dropout Risk Dashboard", "home",
                 f"Accuracy: {acc}% &middot; {total:,} students", body)

# ─── Financial ───────────────────────────────────────────────────────────────
def render_financial() -> str:
    c  = build_financial(DF)
    st = c["stats"]
    stat_cards = (
        _stat("Debtor Dropout %",    f"{st['debtor_dropout']}%",  "students with debt","var(--rd)") +
        _stat("Non-Debtor Dropout %",f"{st['nondbtor_dropout']}%","no debt","var(--gr)") +
        _stat("Fees Late Dropout %", f"{st['feeslate_dropout']}%","tuition overdue","var(--rd)") +
        _stat("Scholar Dropout %",   f"{st['scholar_dropout']}%", "scholarship holders","var(--yw)") +
        _stat("No Scholar Dropout %",f"{st['noscholar_dropout']}%","no scholarship","var(--mu)")
    )
    body = f"""
<div class="sg">{stat_cards}</div>
<div class="cg">
  {_chart_card("fin1",c["Debtor_chart"],"Debtor vs Outcome")}
  {_chart_card("fin2",c["Tuition_fees_up_to_date_chart"],"Tuition Fees vs Outcome")}
</div>
<div class="cg">
  {_chart_card("fin3",c["Scholarship_holder_chart"],"Scholarship vs Outcome")}
  {_chart_card("fin4",c["combo_chart"],"Dropout % by Financial Profile")}
</div>
<h3 style="margin:18px 0 10px;font-size:14px;">Financial Risk Precautions</h3>
{_precautions("Financial", RISK_PRECAUTIONS["Financial"]["precautions"])}"""
    return _page("Financial Status","financial","Financial Risk Factors",body)

# ─── Marital ─────────────────────────────────────────────────────────────────
def render_marital() -> str:
    c  = build_marital(DF)
    st = c["stats"]
    stat_cards = (
        _stat("Most Common Status", st["most_common"], "marital group","var(--bl)") +
        _stat("Highest Risk Group",  st["highest_risk"],"dropout rate","var(--rd)") +
        _stat("Lowest Risk Group",   st["lowest_risk"], "dropout rate","var(--gr)") +
        _stat("Total Students",      f"{len(DF):,}",    "in dataset")
    )
    body = f"""
<div class="sg">{stat_cards}</div>
<div class="cg">
  {_chart_card("mar1",c["grouped"],"Marital Status vs Academic Outcome")}
  {_chart_card("mar2",c["pie"],"Distribution by Marital Status")}
</div>
<div class="cf">
  {_chart_card("mar3",c["dropout_rate"],"Dropout Rate by Marital Status",full=True)}
</div>
<h3 style="margin:18px 0 10px;font-size:14px;">Marital Risk Precautions</h3>
{_precautions("Marital", RISK_PRECAUTIONS["Marital"]["precautions"])}"""
    return _page("Marital Status","marital","Marital Risk Factors",body)

# ─── Demographic ─────────────────────────────────────────────────────────────
def render_demographic() -> str:
    c  = build_demographic(DF)
    st = c["stats"]
    stat_cards = (
        _stat("Female %",       f"{st['pct_female']}%",  "of students","var(--pu)") +
        _stat("Male %",         f"{st['pct_male']}%",    "of students","var(--bl)") +
        _stat("Average Age",    str(st["avg_age"]),       "at enrollment","var(--yw)") +
        _stat("Displaced %",    f"{st['pct_displaced']}%","students displaced","var(--rd)") +
        _stat("International %",f"{st['pct_intl']}%",     "international","var(--cy)")
    )
    body = f"""
<div class="sg">{stat_cards}</div>
<div class="cg">
  {_chart_card("dem1",c["gender_pie"],"Gender Distribution")}
  {_chart_card("dem2",c["gender_target"],"Gender vs Academic Outcome")}
</div>
<div class="cg">
  {_chart_card("dem3",c["age_dropout"],"Dropout Rate by Age Group")}
  {_chart_card("dem4",c["displaced"],"Displaced Students vs Outcome")}
</div>
<h3 style="margin:18px 0 10px;font-size:14px;">Demographic Risk Precautions</h3>
{_precautions("Demographic", RISK_PRECAUTIONS["Demographic"]["precautions"])}"""
    return _page("Demographic Status","demographic","Demographic Risk Factors",body)

# ─── Social ──────────────────────────────────────────────────────────────────
def render_social() -> str:
    c  = build_social(DF)
    st = c["stats"]
    stat_cards = (
        _stat("Avg Unemployment", f"{st['avg_unemp']}%","macro indicator","var(--rd)") +
        _stat("Avg GDP Growth",   f"{st['avg_gdp']}",   "macro indicator","var(--gr)") +
        _stat("Avg Inflation",    f"{st['avg_infl']}%", "macro indicator","var(--yw)") +
        _stat("Special Needs %",  f"{st['pct_special']}%","of students","var(--pu)")
    )
    body = f"""
<div class="sg">{stat_cards}</div>
<div class="cg">
  {_chart_card("soc1",c["unemp"],"Dropout Rate by Unemployment Rate")}
  {_chart_card("soc2",c["gdp"],"Dropout Rate by GDP Environment")}
</div>
<div class="cg">
  {_chart_card("soc3",c["mom_edu"],"Dropout Rate by Mother's Education")}
  {_chart_card("soc4",c["special"],"Special Needs vs Academic Outcome")}
</div>
<h3 style="margin:18px 0 10px;font-size:14px;">Social Risk Precautions</h3>
{_precautions("Social", RISK_PRECAUTIONS["Social"]["precautions"])}"""
    return _page("Social Status","social","Social Risk Factors",body)

# ─── Risk Analysis ───────────────────────────────────────────────────────────
def render_risk() -> str:
    a        = build_risk_analysis(DF)
    max_sc   = a["max_score"]
    rank_cards = ""
    medals   = ["&#129351;","&#129352;","&#129353;","4th","5th"]
    colors   = ["var(--rd)","var(--yw)","var(--gr)","var(--bl)","var(--pu)"]
    for idx,(cat,score) in enumerate(a["sorted_risk"]):
        pct = int(score/max_sc*100)
        rank_cards += f"""<div class="rc">
          <div class="ct">{medals[idx]} Rank #{idx+1}</div>
          <div class="sc2" style="color:{colors[idx]}">{cat}</div>
          <div style="font-size:12px;color:var(--mu)">Score: {score:.4f}</div>
          <div class="rb"><div class="rbf" style="width:{pct}%"></div></div>
        </div>"""

    prec_all = "".join(
        _precautions(cat, RISK_PRECAUTIONS[cat]["precautions"])
        for cat,_ in a["sorted_risk"])

    mm   = a["model_metrics"]
    body = f"""
<div class="rg">{rank_cards}</div>
<div class="cg">
  {_chart_card("risk1",a["risk_chart"],"Risk Contribution by Category")}
  {_chart_card("fi1",a["fi_chart"],"Top 20 Feature Importances")}
</div>
<h3 style="margin:18px 0 10px;font-size:14px;">
  &#9888; Top Risk Category: <span style="color:var(--rd)">{a['top_risk']}</span>
  &nbsp;— Precautions by Priority
</h3>
{prec_all}
<div class="cc" style="margin-top:18px">
  <h3>Model Performance Summary</h3>
  <table>
    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Overall Accuracy</td><td><strong>{mm.get('accuracy',0)}%</strong></td></tr>
      <tr><td>Algorithm</td><td>Random Forest (200 trees, max_depth=12)</td></tr>
      <tr><td>Training Split</td><td>80 / 20</td></tr>
      <tr><td>Features</td><td>{len(FEATURE_NAMES)}</td></tr>
    </tbody>
  </table>
</div>"""
    return _page("Risk Analysis","risk",f"Top Risk: {a['top_risk']}",body)

# ─── Predict ─────────────────────────────────────────────────────────────────
def render_predict(result: dict = None) -> str:
    # Build form fields for top meaningful features
    priority_feats = [
        ("Debtor","Debtor (0=No, 1=Yes)","number","0","0","1"),
        ("Tuition_fees_up_to_date","Tuition Fees Up-to-Date (0=No, 1=Yes)","number","1","0","1"),
        ("Scholarship_holder","Scholarship Holder (0=No, 1=Yes)","number","0","0","1"),
        ("Age_at_enrollment","Age at Enrollment","number","20","15","70"),
        ("Gender","Gender (0=Female, 1=Male)","number","0","0","1"),
        ("Displaced","Displaced (0=No, 1=Yes)","number","0","0","1"),
        ("International","International (0=No, 1=Yes)","number","0","0","1"),
        ("Educational_special_needs","Special Needs (0=No, 1=Yes)","number","0","0","1"),
        ("Admission_grade","Admission Grade","number","130","0","200"),
        ("Previous_qualification_grade","Previous Qualification Grade","number","130","0","200"),
        ("Curricular_units_1st_sem_enrolled","1st Sem Units Enrolled","number","5","0","10"),
        ("Curricular_units_1st_sem_approved","1st Sem Units Approved","number","4","0","10"),
        ("Curricular_units_1st_sem_grade","1st Sem Average Grade","number","13","0","20"),
        ("Curricular_units_2nd_sem_enrolled","2nd Sem Units Enrolled","number","5","0","10"),
        ("Curricular_units_2nd_sem_approved","2nd Sem Units Approved","number","4","0","10"),
        ("Curricular_units_2nd_sem_grade","2nd Sem Average Grade","number","13","0","20"),
        ("Unemployment_rate","Unemployment Rate (%)","number","11","0","25"),
        ("Inflation_rate","Inflation Rate (%)","number","1.5","-5","10"),
        ("GDP","GDP Growth Rate","number","0.5","-5","5"),
        ("Marital_status","Marital Status (1=Single…6=Separated)","number","1","1","6"),
    ]
    # hidden fields for remaining features (set to defaults)
    shown = {f[0] for f in priority_feats}
    hidden_fields = "".join(
        f'<input type="hidden" name="{feat}" value="0"/>'
        for feat in FEATURE_NAMES if feat not in shown)

    form_inputs = ""
    for name,label,ftype,default,fmin,fmax in priority_feats:
        form_inputs += f"""<div class="fgrp">
          <label>{label}</label>
          <input type="{ftype}" name="{name}" value="{default}"
                 min="{fmin}" max="{fmax}" step="any"/>
        </div>"""

    result_html = ""
    if result:
        risk_color = {"High":"var(--rd)","Medium":"var(--yw)","Low":"var(--gr)"}
        rc_color   = risk_color.get(result["risk_level"],"var(--bl)")
        prec_rows  = "".join(
            f'<div class="pi"><div class="pn">{i+1}</div><div>{p}</div></div>'
            for i,p in enumerate(result["precautions"]))
        jdata = result["proba_chart"].replace("\\","\\\\").replace("`","\\`")
        result_html = f"""
<div class="res">
  <div class="hl">Prediction: <span style="color:{rc_color}">{result['prediction']}</span></div>
  <div style="color:var(--mu);font-size:13px">Dropout Probability: {result['dropout_prob']}%</div>
  <div class="ig">
    <div class="ii"><div class="k">Risk Level</div>
      <div class="v" style="color:{rc_color}">{result['risk_level']}</div></div>
    <div class="ii"><div class="k">Dropout Probability</div>
      <div class="v">{result['dropout_prob']}%</div></div>
    <div class="ii"><div class="k">Top Risk Category</div>
      <div class="v" style="color:var(--yw)">{result['top_category']}</div></div>
    <div class="ii"><div class="k">Prediction</div>
      <div class="v">{result['prediction']}</div></div>
  </div>
  <div id="proba_chart"></div>
  <script>rc('proba_chart',`{jdata}`);</script>
</div>
<div class="ps">
  <h3>&#9989; Recommended Precautions ({result['top_category']} Risk)</h3>
  {prec_rows}
</div>"""

    body = f"""
{result_html}
<div class="cf">
  <h3>Student Risk Predictor — Enter Student Details</h3>
  <form method="POST" action="/predict">
    {hidden_fields}
    <div class="fg">{form_inputs}</div>
    <button type="submit" class="btn">&#128302; Predict Dropout Risk</button>
  </form>
</div>"""
    return _page("Predict Student","predict","ML Risk Predictor",body)

# ─────────────────────────────────────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET","POST"])
def home():
    result = None
    if request.method == "POST":
        result = predict_single(request.form.to_dict())
    return render_home(predict_result=result)

@app.route("/financial")
def financial():
    return render_financial()

@app.route("/marital")
def marital():
    return render_marital()

@app.route("/demographic")
def demographic():
    return render_demographic()

@app.route("/social")
def social():
    return render_social()

@app.route("/risk")
def risk():
    return render_risk()

@app.route("/predict", methods=["GET","POST"])
def predict():
    result = None
    if request.method == "POST":
        result = predict_single(request.form.to_dict())
    return render_predict(result)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data   = request.get_json(force=True)
    result = predict_single(data)
    return jsonify({k:v for k,v in result.items() if k != "proba_chart"})

@app.route("/api/overview")
def api_overview():
    return jsonify({
        "total":     len(DF),
        "dropouts":  int((DF["Target"]=="Dropout").sum()),
        "graduates": int((DF["Target"]=="Graduate").sum()),
        "enrolled":  int((DF["Target"]=="Enrolled").sum()),
        "accuracy":  MODEL_METRICS.get("accuracy",0),
        "features":  len(FEATURE_NAMES),
    })

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap — safe for Gunicorn multi-worker fork model
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap():
    """Load data and train model. Thread-safe; runs once per worker process."""
    global DF, _bootstrapped
    with _bootstrap_lock:
        if _bootstrapped:
            return
        # FIX 2 cont.: clear any stale .pyc that causes line-number mismatches
        import glob as _glob
        for _pyc in _glob.glob(os.path.join(os.path.dirname(__file__),
                                             "__pycache__", "*.pyc")):
            try:
                os.remove(_pyc)
            except OSError:
                pass

        os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)
        DF = load_data()
        train_model(DF)
        _bootstrapped = True
        print(f"[OK] {len(DF):,} records loaded  |  "
              f"Model accuracy: {MODEL_METRICS['accuracy']}%  |  "
              f"Classes: {MODEL_METRICS['classes']}")
        print("[  ] Dashboard running")


# FIX 3: Use before_request guard instead of @before_first_request
# (@before_first_request was removed in Flask 3.x).
# This runs bootstrap() lazily on the very first request in each worker,
# which is the correct pattern for Gunicorn pre-fork workers.
@app.before_request
def ensure_bootstrapped():
    if not _bootstrapped:
        bootstrap()


if __name__ == "__main__":
    bootstrap()
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") != "production"
    app.run(debug=debug, host="0.0.0.0", port=port, use_reloader=False)
