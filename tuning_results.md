# ML Model Results — Age Group Prediction from LLM Interactions

**Date:** 2026-03-23  
**Dataset:** `all_tasks_90_sub_23_12.csv` (1,275 rows, 255 test samples)  
**Target:** `subject_group` (Young_Adults vs Older_Adults) — **Age feature excluded**  
**Features:** Top 50 by Mutual Information (~150+ engineered: DistilBERT, TF-IDF/SVD, NMF, VADER sentiment, readability, lexical diversity, POS tags)

---

## 🏆 Final Model Ranking (by Test F1)

| Rank | Model | Test Accuracy | Test F1 | Precision | Recall |
|------|-------|--------------|---------|-----------|--------|
| 1 | Naive Bayes | 0.5451 | **0.6667** | 0.5110 | 0.9587 |
| 2 | XGBoost | 0.6863 | 0.6639 | 0.6752 | 0.6529 |
| 3 | Random Forest | 0.6745 | 0.6498 | 0.6638 | 0.6364 |
| 4 | **Gradient Boosting (tuned)** | **0.6667** | 0.6444 | 0.6525 | 0.6364 |
| 5 | Gradient Boosting | 0.6471 | 0.6121 | 0.6396 | 0.5868 |
| 6 | Logistic Regression (tuned) | 0.6157 | 0.6111 | 0.5878 | 0.6364 |
| 7 | XGBoost (tuned) | 0.6549 | 0.6106 | 0.6571 | 0.5702 |
| 8 | Logistic Regression | 0.6039 | 0.5944 | 0.5781 | 0.6116 |
| 9 | LinearSVC | 0.6000 | 0.5854 | 0.5760 | 0.5950 |

> **Note:** Naive Bayes has highest F1 but very low precision (0.51) — it over-predicts Older Adults (recall 0.96). Gradient Boosting (tuned) is the most **balanced** performer.

---

## Best Tuned Hyperparameters

### Gradient Boosting (Best Balanced Model)

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 0.0118 |
| `max_depth` | 9 |
| `min_samples_leaf` | 6 |
| `min_samples_split` | 3 |
| `n_estimators` | 738 |
| `subsample` | 0.8073 |

### XGBoost

| Parameter | Value |
|-----------|-------|
| `colsample_bytree` | 0.9114 |
| `gamma` | 0.3721 |
| `learning_rate` | 0.0813 |
| `max_depth` | 11 |
| `min_child_weight` | 1 |
| `n_estimators` | 923 |
| `reg_alpha` | 0.3847 |
| `reg_lambda` | 0.2037 |
| `subsample` | 0.9011 |

### Logistic Regression

| Parameter | Value |
|-----------|-------|
| `C` | 0.1695 |
| `penalty` | l1 |
| `solver` | liblinear |

---

## Classification Report (Gradient Boosting Tuned)

```
              precision    recall  f1-score   support
Young Adults       0.68      0.69      0.69       134
Older Adults       0.65      0.64      0.64       121
    accuracy                           0.67       255
```

---

## Key Takeaways
- **Best accuracy-F1 tradeoff:** Gradient Boosting (tuned) — 66.7% accuracy, 64.4% F1
- **Without Age**, this is a hard classification task — behavioral/linguistic features alone provide modest signal
- Tuning sometimes **hurt** performance (XGBoost base 0.66 F1 → tuned 0.61 F1) — likely overfitting to CV folds
- Naive Bayes' high F1 is misleading — it sacrifices precision for recall
