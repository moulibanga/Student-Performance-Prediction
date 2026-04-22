# Student Performance Prediction from Survey Data

## Overview
This project analyzes survey data from students in a university data science course to determine whether study habits and lifestyle factors can predict academic performance.

Using cleaned survey responses, I built multiple machine learning models to predict:
- Pass/fail outcomes
- Letter grades (A–F)
- Numeric percentage grades

## Dataset
- Source: Student survey responses + exam scores
- Features include:
  - Study habits
  - Sleep patterns
  - Lecture attendance
  - Exam perception
- Data preprocessing:
  - Removed invalid or missing score entries
  - Converted categorical features using one-hot encoding
  - Filled missing values with column means
  - Engineered labels for pass/fail and letter grades

## Models
### Classification
- Logistic Regression
- K-Nearest Neighbors (k=5)
- Decision Tree
- Random Forest

### Regression
- Linear Regression

## Results
- **Pass/Fail Classification**
  - Decision Tree accuracy: ~73.7%
  - Logistic Regression accuracy: ~68.4%
  - KNN accuracy: ~63.2%

- **Letter Grade Classification**
  - Random Forest accuracy: ~47.4%
  - Logistic Regression accuracy: ~36.8%
  - KNN accuracy: ~21.1%

- **Regression**
  - Linear Regression MAE: ~9.86 percentage points

## Key Insight
Models performed significantly better when predicting coarse outcomes (pass/fail) compared to fine-grained predictions (exact letter grades). This suggests that survey-based features capture general performance trends but struggle to distinguish nuanced academic differences.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- NumPy
- Matplotlib

## Files
- `student_performance_analysis.ipynb` — full data pipeline and modeling
- `report.pdf` — detailed analysis and results

## How to Run
1. Open the notebook in Jupyter or Google Colab
2. Run all cells
3. Models will train and output evaluation metrics
