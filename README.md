# 🎓 Academic Dropout Risk Dashboard

A **single-file Python web application** that loads the **UCI "Predict Students' Dropout and Academic Success"** dataset, trains a **Random Forest** classifier, and serves a fully interactive dark-themed dashboard — entirely in Python. The Flask backend handles routing and ML inference; the entire frontend (HTML, CSS, JavaScript) is generated as Python f-strings. Zero external template files.

---

## 📂 Project Structure

```
academic_dropout/
├── Harshit_Academic_DropoutAnalysis.py  ← Single file: backend + frontend (all Python)
├── requirements.txt                     ← Python dependencies
├── render.yaml                          ← Render.com deployment config
├── README.md                            ← This file
└── data/
    └── dataset.csv                      ← (optional) Place the real UCI CSV here
```

> **No templates directory.** All HTML, CSS, and JavaScript are generated as Python f-strings inside `Harshit_Academic_DropoutAnalysis.py`.

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| **Name** | Predict Students' Dropout and Academic Success |
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success |
| **Records** | 4,424 students |
| **Features** | 36 (demographic, socioeconomic, academic, macroeconomic) |
| **Target** | `Dropout` / `Enrolled` / `Graduate` |
| **Format** | CSV, semicolon-separated |

### Key Feature Groups

| Group | Features |
|-------|----------|
| **Financial** | `Debtor`, `Tuition_fees_up_to_date`, `Scholarship_holder` |
| **Marital** | `Marital_status` (1=Single … 6=Legally Separated) |
| **Demographic** | `Age_at_enrollment`, `Gender`, `Nationality`, `Displaced`, `International` |
| **Social** | `Unemployment_rate`, `GDP`, `Inflation_rate`, `Mothers_qualification`, `Fathers_qualification`, `Educational_special_needs` |
| **Academic** | `Curricular_units_*_sem_approved/grade`, `Admission_grade`, `Previous_qualification_grade` |

> **No dataset download required to run.** If `data/dataset.csv` is absent, the app auto-generates a 4,424-record synthetic dataset with identical statistical properties.

---

## 🛠️ Technologies Used

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.9+ (3.11 recommended for Render deployment) |
| **Web Framework** | Flask 3.x |
| **Production Server** | Gunicorn 21.x (WSGI server for Render / cloud deployment) |
| **Machine Learning** | scikit-learn — RandomForestClassifier, StandardScaler, LabelEncoder |
| **Data Processing** | pandas, numpy |
| **Charts** | Plotly (Python → JSON → rendered by Plotly.js in browser) |
| **Frontend** | Pure Python f-strings generating HTML5 / CSS3 / JS (no templates) |
| **Concurrency** | `threading.Lock` — thread-safe lazy bootstrap for Gunicorn workers |
| **Serialisation** | joblib |

---

## ⚙️ Setup & Installation

### 1. Clone / copy the project
```bash
git clone <repo-url>
cd academic_dropout
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Add the real UCI dataset
Download from the UCI link above, rename to `dataset.csv`, place in the `data/` folder:
```
academic_dropout/data/dataset.csv
```
If absent, a 4,424-record synthetic dataset is generated automatically on first request.

---

## ▶️ Run Locally

```bash
python Harshit_Academic_DropoutAnalysis.py
```

Expected console output:
```
[OK] 4,424 records loaded  |  Model accuracy: XX.XX%  |  Classes: ['Dropout', 'Enrolled', 'Graduate']
[  ] Dashboard running
```

Open your browser at **http://127.0.0.1:5000**

---

## 🌐 Deploy to Render (free hosting)

1. Push the project folder to a **GitHub repository**
2. Go to [https://render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — no manual config needed
5. Click **Deploy** → get a live public URL in ~3 minutes

**Manual Render settings** (if not using `render.yaml`):

| Setting | Value |
|---------|-------|
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn Harshit_Academic_DropoutAnalysis:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT` |
| **Python Version** | `3.11.0` |

---

## 🖥️ Dashboard — All Features at One URL

The entire dashboard is served at **`http://127.0.0.1:5000`** (or your live Render URL) as a **single-page tabbed website**. No separate URLs needed.

| Tab | Content |
|-----|---------|
| 🏠 **Overview** | Total students, dropout %, model accuracy, outcome pie chart, confusion matrix, feature importances, classification report |
| 💰 **Financial** | Debtor/fees/scholarship charts, dropout % by financial profile, precautions |
| 💍 **Marital** | Marital group distribution, grouped bar, dropout rate per status, precautions |
| 🌍 **Demographic** | Gender pie, gender vs outcome, age-group dropout rate, displaced students, precautions |
| 🤝 **Social** | Unemployment/GDP/inflation charts, mother's education, special needs, precautions |
| ⚠️ **Risk Analysis** | Ranked risk categories, top 20 feature importances, all 25 precautions by priority |
| 🔮 **Predict Student** | Fill in a student profile → dropout probability, risk level, top risk category, tailored precautions — all inline |

### REST API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/overview` | JSON summary of dataset & model metrics |
| `POST` | `/api/predict` | JSON body → prediction, dropout_prob, risk_level, top_category |

---

## 🔮 ML Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Trees | 200 (`n_estimators=200`) |
| Max depth | 12 (`max_depth=12`) |
| Train/Test split | 80 % / 20 % stratified |
| Scaling | StandardScaler (zero mean, unit variance) |
| Label encoding | LabelEncoder (`Dropout=0, Enrolled=1, Graduate=2`) |
| Random seed | 42 |
| Accuracy | ~89% on held-out test set |
| Risk thresholds | Low < 35 % · Medium 35–60 % · High > 60 % |

---

## 🐛 Render / Gunicorn Deployment Fixes Applied

The following production bugs were diagnosed and fixed in the current version:

| # | Error | Root Cause | Fix Applied |
|---|-------|-----------|-------------|
| 1 | `TypeError: ignore() got an unexpected keyword argument 'type'` | `warnings.filterwarnings("ignore")` at **module level** conflicts with Gunicorn's internal warning filter on Python 3.12+ | Removed module-level call; `warnings` imported **locally inside `train_model()`** only, scoped to `UserWarning` and `FutureWarning` |
| 2 | Traceback points to line 1377 but file is 1240 lines | Stale `__pycache__/*.pyc` from a previously longer build cached on Render | `bootstrap()` purges `__pycache__` on first run |
| 3 | `AttributeError: 'NoneType' has no attribute ...` on GET / | `bootstrap()` only ran under `if __name__ == "__main__"` — never executed by Gunicorn workers | `@app.before_request` lazy guard calls `bootstrap()` on first request per worker; `threading.Lock` prevents race conditions |

---

## ⚠️ Risk Categories & Precautions

The app ranks **five status categories** by normalised contribution to dropout risk:

1. **Financial** — debt, tuition fees, scholarships *(highest risk)*
2. **Academic** — approved units, grades, admission scores
3. **Social** — macroeconomic indicators, parental education, special needs
4. **Demographic** — age, gender, displacement, international status
5. **Marital** — marital/family status *(lowest risk)*

Each category includes **5 evidence-based precautions** shown on the dashboard and in prediction results.

---

## 📋 Key Information

- **Single file**: `Harshit_Academic_DropoutAnalysis.py` — ~1,280 lines covering data generation, ML training, Plotly chart building, HTML/CSS/JS rendering, and Flask routing. No external template files.
- **Gunicorn-safe**: Lazy `@app.before_request` bootstrap pattern works correctly with Gunicorn's pre-fork worker model. `threading.Lock` ensures thread safety.
- **Offline-capable**: Runs without internet except for Plotly.js (loaded from CDN). Swap for a local copy for full offline use.
- **Synthetic data fallback**: No dataset download required — app generates realistic data automatically.
- **Dark UI**: Professional dark-themed dashboard with responsive grid layout and tab navigation.

---

## 📄 License

MIT License — free to use, modify, and distribute.
