 # 🎓 Academic Dropout Risk Dashboard

A single-file Python web application that loads the **UCI "Predict Students' Dropout and Academic Success"** dataset, trains a **Random Forest** classifier, and serves an interactive dark-themed dashboard — entirely in Python (Flask backend + Python f-string HTML frontend, zero external template files).

---

## 📂 Project Structure

```
academic_dropout/
├── app.py            ← Single file: backend + frontend (all Python)
├── requirements.txt  ← Python dependencies
├── README.md         ← This file
└── data/
    └── dataset.csv   ← (optional) Place the real UCI CSV here
```

> **No templates directory.** All HTML, CSS, and JavaScript are generated as Python strings inside `app.py`.

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

> **No dataset download required to run.** If `data/dataset.csv` is absent, the app auto-generates a synthetic dataset with identical statistical properties.

---

## 🛠️ Technologies Used

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.9+ |
| **Web Framework** | Flask 3.x |
| **Machine Learning** | scikit-learn (RandomForestClassifier, StandardScaler, LabelEncoder) |
| **Data Processing** | pandas, numpy |
| **Charts** | Plotly (Python → JSON → rendered by Plotly.js in browser) |
| **Frontend** | Pure Python f-strings generating HTML5 / CSS3 / JS (no templates) |
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
Download the CSV from the UCI link above, rename it `dataset.csv`, and place it in the `data/` folder:
```
academic_dropout/data/dataset.csv
```
If absent, a 4,424-record synthetic dataset is generated automatically.

---

## ▶️ Run the Application

```bash
python app.py
```

Expected console output:
```
[OK] 4,424 records loaded  |  Model accuracy: XX.XX%  |  Classes: ['Dropout', 'Enrolled', 'Graduate']
[  ] Dashboard: http://127.0.0.1:5000
```

Open your browser at **http://127.0.0.1:5000**

---

## 🖥️ Dashboard Pages

| URL | Page | Description |
|-----|------|-------------|
| `/` | **Home** | Overall stats, outcome distribution pie chart, confusion matrix, feature importances, classification report |
| `/financial` | **Financial Status** | Debtor/fees/scholarship analysis, dropout % by financial profile, precautions |
| `/marital` | **Marital Status** | Marital group distribution, dropout rate per group, precautions |
| `/demographic` | **Demographic Status** | Gender, age groups, displaced students, international students, precautions |
| `/social` | **Social Status** | Unemployment rate, GDP environment, parents' education, special needs, precautions |
| `/risk` | **Risk Analysis** | Ranked risk categories, top 20 feature importances, all precautions by priority |
| `/predict` | **Predict Student** | Enter a student profile → get dropout probability, risk level, top category & tailored precautions |

### REST API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/api/overview` | JSON summary of dataset & model metrics |
| `POST` | `/api/predict` | JSON body → dropout prediction response |

---

## 🔮 ML Model Details

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Classifier |
| Trees | 200 (`n_estimators=200`) |
| Max depth | 12 |
| Train/Test split | 80 % / 20 % |
| Scaling | StandardScaler (zero mean, unit variance) |
| Label encoding | LabelEncoder (`Dropout=0, Enrolled=1, Graduate=2`) |
| Random seed | 42 |

---

## ⚠️ Risk Categories & Precautions

The app ranks **five status categories** by their normalised contribution to dropout risk:

1. **Financial** — debt, tuition fees, scholarships
2. **Academic** — approved units, grades, admission scores
3. **Demographic** — age, gender, displacement, international status
4. **Social** — macroeconomic indicators, parental education, special needs
5. **Marital** — marital/family status

Each category includes **5 evidence-based precautions** displayed on the dashboard and in prediction results.

---

## 📋 Key Information

- **Single file**: `app.py` contains ~500 lines of Python covering data generation, ML training, chart generation, HTML/CSS/JS rendering, and Flask routing — no external template files needed.
- **Offline-capable**: Runs fully without internet (Plotly.js is loaded from CDN; swap for a local copy for full offline use).
- **Extensible**: Drop in a real CSV, swap the classifier, or add new routes — all in one file.
- **Dark UI**: Professional dark-themed dashboard with responsive grid layout.

---

## 📄 License

MIT License — free to use, modify, and distribute.
