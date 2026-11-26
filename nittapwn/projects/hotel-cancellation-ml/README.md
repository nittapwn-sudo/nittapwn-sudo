# 🏨 Hotel Booking Cancellation Prediction with Feature Selection

> **Research paper** — *Optimizing Hotel Cancellation Trends Analysis using Machine Learning Models with the Feature Selection Method*
> Submitted to the 12th Asia Undergraduate Conference on Computing (AUC2), 2024
> 📄 [Read the full paper (PDF, Thai with English abstract)](paper/Hotel-Cancellation-Feature-Selection-Paper.pdf)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)

---

## 📌 Problem
Guests often cancel or don't show up, leaving hotels with empty rooms they could have sold. Hotels respond with **overbooking**, which is risky when fewer guests cancel than expected.
This project studies **which booking factors drive cancellations** and compares classification models for predicting them, to support better overbooking decisions.

## 📊 Dataset
[Hotel Reservations Dataset (Kaggle)](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset) — **36,275 bookings, 19 columns**
(lead time, arrival date, number of guests, meal plan, room type, market segment, average room price, special requests, previous cancellations, booking status, …)

## 🔧 Method
1. **Preprocessing** — encoded categorical columns with `LabelEncoder`
2. **Exploratory analysis** — correlation matrix and heatmaps between all features and against `booking_status`
3. **Outlier removal** — dropped rows outside the IQR → **27,318 rows, 18 columns** remain
4. **Train/test split** — 67% / 33%, the same split for every model
5. **Models** — Decision Tree (Gini, no max depth) and Logistic Regression (default parameters)
6. **Feature selection** — correlation-based filter: kept the top 5 positively and top 5 negatively correlated features
7. **Evaluation** — Accuracy, Recall, Precision, F1 on the test set

## 📈 Results

| Model | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| Decision Tree (all features) | 86% | 80% | 79% | 79% |
| Logistic Regression (all features) | 79% | 58% | 73% | 64% |
| Decision Tree + correlation feature selection | 100% | 100% | 100% | 100% |
| Logistic Regression + correlation feature selection | 100% | 100% | 100% | 100% |

### 💡 Key insights for the business
- **Lead time** is the strongest factor (correlation ≈ −0.44 with a kept booking): the earlier a guest books, the more likely they cancel.
- **Number of special requests** (≈ +0.25) and **repeated guests** (≈ +0.11) are linked to bookings that are kept.
- Hotels could use these signals to decide **how much to overbook** and which bookings to follow up on.

### ⚠️ Limitations & next steps
A perfect 100% score is unusual for this kind of problem and **likely points to data leakage or overfitting** rather than a perfect model — the paper also notes the dataset may be limited. Next steps:
- Re-check the selected features for anything derived from the target
- Validate with **k-fold cross-validation** and a held-out set split *before* outlier removal
- Compare with stronger models (Random Forest, Gradient Boosting) and report ROC-AUC

## 👥 Authors
Nittha Phuwananwasusin, Onchuma Triyaphatthanakamjorn, Kriangsak Jeklek — B.Sc. Information Technology, Faculty of Science, Silpakorn University
Advisor: Dr. Panyanat Aonpong, Department of Computing, Faculty of Science, Silpakorn University
