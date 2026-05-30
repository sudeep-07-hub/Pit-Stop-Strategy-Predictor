# 🏎️ F1 Pit Stop Strategy Predictor

> A machine learning model that predicts whether an F1 driver will pit on the next lap,  
> built on real telemetry data from the 2023 Formula 1 season.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

Pit stop timing is one of the most critical decisions in Formula 1 racing. Strategy engineers monitor live telemetry every lap and decide — sometimes with seconds to spare — whether to bring a driver in. This project replicates that decision-making process using machine learning.

Given the current race state (tyre age, lap time trends, position, laps remaining), the model predicts:

```
0 → Driver will NOT pit next lap
1 → Driver WILL pit next lap
```

This is a **binary classification problem** with severe class imbalance (~97% non-pit laps), making it a realistic and challenging industry-grade ML task.

---

## 📊 Dataset

- **Source:** [FastF1 Python Library](https://github.com/theOehrly/Fast-F1) — official F1 telemetry API
- **Season:** 2023 Formula 1 World Championship
- **Races used:** 16 races (Bahrain through United States)
- **Total laps:** ~13,500 clean racing laps
- **Train/Test split:** Time-based — trained on 14 races, tested on Britain + Italy

| Split | Races | Total Laps | Pit Stop Laps |
|-------|-------|-----------|---------------|
| Train | 14 races (Bahrain → Singapore) | ~11,892 | ~210 |
| Test  | 2 races (Britain + Italy)       | ~1,692  | 47   |

### Data Quality Notes
- TSU (Tsunoda) Canada 2023 — TyreLife telemetry missing across ~20 laps due to fastf1 recording gap. Rows dropped (0.26% of dataset).
- Safety car and VSC laps removed via `pick_quicklaps()` — verified with `TrackStatus.value_counts()`.
- Lap 1 removed from all races — standing start conditions are not representative of race pace.

---

## ⚙️ Feature Engineering

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `TyreLife` | Age of current tyre in laps | Primary pit stop trigger — older tyres = more likely to pit |
| `TyreLifeSquared` | TyreLife² | Captures non-linear degradation acceleration |
| `LapTimeSeconds` | Lap time converted to float seconds | Raw pace indicator |
| `LapTimeDelta` | Current lap time minus driver's average for that race | Detects sudden slowdowns due to tyre wear |
| `LapsRemaining` | Total laps − current lap number | Strategic window — teams pit when there are enough laps to benefit |
| `Position` | Current race position | Position influences pit strategy (undercut/overcut) |

### Target Variable Construction
```python
# Look-ahead labeling — created BEFORE cleaning to preserve pit laps
laps['will_pit_next_lap'] = (
    laps.groupby('Driver')['PitInTime']
    .shift(-1)
    .notna()
    .astype(int)
)
```
The target is constructed **before** `pick_quicklaps()` cleaning — pit laps are slow and would otherwise be removed, hiding the very signal we want to predict.

---

## 🧠 Modelling

### Class Imbalance
```
Class 0 (no pit) : 96.6%
Class 1 (pit)    :  3.4%
```
Handled using `class_weight='balanced'` for Logistic Regression and Random Forest, and `scale_pos_weight` for XGBoost. This penalises the model more heavily for missing a pit stop than for a false alarm.

### Models Trained

| Model | Precision (pit) | Recall (pit) | F1 (pit) | Accuracy |
|-------|----------------|--------------|----------|----------|
| Logistic Regression | 0.07 | **0.85** | 0.12 | 0.66 |
| Random Forest | 0.00 | 0.00 | 0.00 | 0.97 |
| XGBoost | 0.08 | 0.15 | 0.10 | 0.91 |

### Why Logistic Regression Won
XGBoost and Random Forest are powerful models but require hundreds of minority class examples to learn complex non-linear boundaries. With only ~210 training pit stops, a simple linear decision boundary (LR) generalised better. This is a common real-world finding — **model complexity must match data availability.**

> Random Forest's 97% accuracy is misleading — it simply predicted "no pit stop" for every lap.  
> Accuracy is a lying metric for imbalanced problems. Recall is what matters here.

### Threshold Optimisation
Default threshold of 0.5 was too conservative. Testing across thresholds:

| Threshold | Precision | Recall | F1 |
|-----------|-----------|--------|----|
| 0.5 | 0.06 | 0.15 | 0.08 |
| 0.3 | 0.06 | 0.28 | 0.10 |
| 0.2 | 0.07 | 0.43 | 0.12 |
| 0.1 | 0.07 | 0.62 | 0.12 |

**Selected threshold: 0.3** — balances recall improvement against false alarm rate.

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to interpret the Logistic Regression model globally and per-prediction.

### Global Feature Importance (Summary Plot)
```
Rank 1 → TyreLife        Most influential — old tyres drive pit decisions
Rank 2 → LapTimeDelta    Lap time deterioration is secondary signal
Rank 3 → TyreLifeSquared Partially cancels TyreLife (multicollinearity)
Rank 4 → LapsRemaining   Minor strategic signal
```

### Single Prediction Explanation (Waterfall Plot)
For a correctly predicted pit stop (TyreLife = 32, LapsRemaining = 20):
```
Base value (average)  : -0.444
TyreLife = 32         : +2.51  ← dominant signal
TyreLifeSquared = 1024: -1.64  ← partial correction
LapTimeDelta = 0.146  : +0.18  ← lap getting slower
LapsRemaining = 20    : +0.08  ← strategic window open
─────────────────────────────
Final output f(x)     :  0.68  → PREDICT PIT ✅
```

**Key insight:** A driver on 32-lap-old tyres is the overwhelming predictor. Everything else provides secondary confirmation — matching real F1 strategy logic exactly.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/f1-pitstop-predictor.git
cd f1-pitstop-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook notebook/model.ipynb
```

The notebook will download race data automatically via fastf1 on first run. Subsequent runs use the local cache.

---

## 📁 Project Structure

```
f1-pitstop-predictor/
│
├── notebook/
│   └── model.ipynb          ← full pipeline: data → features → model → SHAP
│
├── data/
│   └── f1_cache/            ← fastf1 local cache (gitignored)
│
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
fastf1==3.3.9
numpy==1.26.4
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=1.7.0
shap>=0.44.0
matplotlib>=3.7.0
jupyter
```

---

## 💡 Key Learnings

- **Data leakage prevention:** Time-based train/test split ensures the model never sees future race data during training — a critical but often overlooked step in sports ML
- **Target variable design:** Building the look-ahead label before cleaning preserves the pit lap signal that would otherwise be removed
- **Imbalanced classification:** Accuracy is meaningless here — recall and F1 on the minority class are the real metrics
- **Model selection:** Simpler models can outperform complex ones when minority class examples are scarce
- **Explainability:** SHAP waterfall plots confirm the model learned genuine racing logic, not spurious correlations

---

## 🔮 Future Improvements

- **More data:** Expand to 3–5 seasons (2019–2023) for thousands more pit stop examples — expected to significantly improve XGBoost performance
- **Additional features:** Gap to car ahead/behind (undercut opportunity), safety car probability, weather conditions
- **Feature refinement:** Remove `TyreLifeSquared` or apply PCA to resolve multicollinearity with `TyreLife`
- **Live inference:** Deploy model as a REST API that accepts live lap telemetry and returns pit probability in real time

---

## 👤 Author

**Sukesh**  
Data Science Student | F1 Enthusiast  
[GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE)

---

## 📄 License

This project is licensed under the MIT License.  
F1 data is sourced via the FastF1 library for educational purposes only.

---

*Week 2 of a 4-week F1 Data Science Portfolio Project*  
*Week 1: [F1 Race Performance Dashboard](https://github.com/YOUR_USERNAME/f1-dashboard)*