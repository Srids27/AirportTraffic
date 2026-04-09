[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Srids27/AirportTraffic/blob/main/AirportTrafficData.ipynb)

# ✈️ Airport Traffic Data Analysis

A machine learning pipeline for analysing Australian airport passenger and aircraft movement data. The notebook performs **three core analytical tasks**: unsupervised clustering, passenger growth prediction, and anomaly detection.

---

## 📋 Overview

This project uses monthly passenger (`mon_pax_web.csv`) and aircraft movement (`mon_acm_web.csv`) data to extract business insights about airport performance trends, forecast future growth, and flag unusual traffic events.

---

## 🗂️ Data Requirements

| File | Description |
|------|-------------|
| `mon_pax_web.csv` | Monthly passenger data per airport (domestic & international, inbound & outbound) |
| `mon_acm_web.csv` | Monthly aircraft movement data per airport |

**Key columns used:**
- `AIRPORT` — Airport name
- `Year`, `Month` — Time period
- `Pax_Total` — Total passenger count
- `Acm_Total` — Total aircraft movements

> ⚠️ The notebook was developed in **Google Colab** and uses `google.colab.files.upload()` to load CSVs at runtime.

---

## 🔬 Analyses

### 1. 🏷️ K-Means Clustering

**Goal:** Group airports into performance categories — *High-growth*, *Stable*, or *Declining* — based on passenger and aircraft movement trends.

**Approach:**
- Aggregate per-airport features: total passengers, average passengers, volatility, CAGR-style growth rate (for both passengers and aircraft movements)
- Scale features using `StandardScaler`
- Apply K-Means clustering (`k=3`)
- Assign business labels based on cluster-average growth metrics

**Evaluation Metrics:**
| Metric | Value |
|--------|-------|
| Silhouette Score | 0.388 |
| Davies-Bouldin Index | 0.523 |
| Calinski-Harabasz Index | 46.491 |

**Output:** `airport_clusters.csv`

---

### 2. 📈 Regression — Passenger Growth Prediction

**Goal:** Predict monthly passenger growth rate using aircraft movements, lag features, and seasonality.

**Features Used:**
- `Acm_Total` — Total aircraft movements
- `lag_1` — Previous month's passenger total
- `lag_12` — Same month in the previous year
- `month`, `quarter` — Seasonality indicators

**Model:** Random Forest Regressor (`n_estimators=200`)

**Train/Test Split:** Last 12 months per airport held out as test set.

**Results:**
| Metric | Value |
|--------|-------|
| MAE | 0.0373 |
| RMSE | 0.0514 |
| R² | 0.7528 |

**Visualisations produced:**
- Feature importance bar chart
- Actual vs. predicted growth line plot
- Residual scatter plot

---

### 3. 🚨 Anomaly Detection

**Goal:** Identify unusual traffic events such as COVID-era drops, strikes, or sudden spikes.

**Two complementary methods:**

| Method | Description |
|--------|-------------|
| **Residual-based** | Flags points where `\|actual − predicted\| > 2σ` (using Random Forest residuals) |
| **Isolation Forest** | Unsupervised outlier detection on the feature space (`contamination=0.05`) |

**Evaluation (no ground-truth labels available):**
| Metric | Value |
|--------|-------|
| Residual anomalies flagged | 2.04 % |
| Isolation Forest anomalies flagged | 5.00 % |
| Overlap (both methods agree) | 1.65 % |

> If ground-truth anomaly labels exist in a column named `anomaly_label`, the notebook will automatically run supervised evaluation (Precision, Recall, F1, Confusion Matrix).

**Output:** `airport_anomalies.csv` — all rows flagged by either method.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Running in Google Colab

1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Run each cell sequentially.
3. When prompted, upload `mon_pax_web.csv` and `mon_acm_web.csv`.

### Running Locally

Replace the Colab upload widgets with local file reads:

```python
# Replace:
# uploaded = files.upload()
# pax = pd.read_csv("mon_pax_web.csv")

# With:
pax = pd.read_csv("/path/to/mon_pax_web.csv")
acm = pd.read_csv("/path/to/mon_acm_web.csv")
```

---

## 📁 Output Files

| File | Contents |
|------|----------|
| `airport_clusters.csv` | Per-airport cluster label and growth features |
| `airport_anomalies.csv` | Rows flagged as anomalous by either detection method |

---

## 🛠️ Tech Stack

- **Python 3**
- **pandas** — data wrangling
- **NumPy** — numerical operations
- **scikit-learn** — K-Means, Random Forest, Isolation Forest, StandardScaler, evaluation metrics
- **Matplotlib** — visualisation

---

## 📌 Notes

- The CAGR-style growth function computes annualised growth as: `((last / first) ^ (12 / n_months)) − 1`
- Airports with fewer than 12 months of data are excluded from the regression train/test split
- All three analyses independently reload the data via Colab's upload widget; when converting to a standalone script, load the data once and pass DataFrames through

---

## 📄 License

This project is for educational and analytical purposes.
