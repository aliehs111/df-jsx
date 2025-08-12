df.jsx – App Overview & Guide
Purpose:df.jsx is a full-stack web app for uploading, cleaning, exploring, and modeling datasets.It helps non-technical and technical users alike understand their data and prepare it for analysis or machine learning.

Core Features

Dataset Upload

Upload CSV files.
Preview the first few rows before saving.
Decide whether to proceed to cleaning or discard the upload.

My Datasets

View all datasets saved to your account.
See metadata such as number of rows, columns, and last updated date.
Click a dataset to view its details.

Dataset Detail

View dataset structure (rows, columns, missing values).
Run info/stats (shape, head, describe).
Clean data:
Handle missing values (drop, fill with mean/median/mode/custom).
Scale features (normalize or standardize).
Encode categorical variables.

Generate visualizations such as correlation heatmaps.
Save cleaned datasets for modeling.

Models

Run pre-trained or built-in models on cleaned datasets.
Supported models include:
Random Forest
Logistic Regression
PCA + KMeans Clustering
Sentiment Analysis (DistilBERT)
Anomaly Detection
Feature Importance

View structured results in a clean UI.

Dashboard

Quick KPI cards (datasets, rows, columns, last upload).
Recent Datasets table.
Quick Actions for Upload, View All Datasets, Run a Model.

Databot

Global chatbot that can answer questions about the app.
On the dashboard or models page, Databot can act as a welcome guide.
On dataset or predictor pages, it can give contextual tips.

Typical Workflow

Upload a CSV file.
Preview the data and decide whether to save it.
Clean and preprocess the dataset.
Save the cleaned version.
Run one or more models to analyze the data.
Review results and download outputs if needed.

Tips

You must be logged in to use the app.
Data cleaning steps are saved so you can revisit them later.
Use Databot for guidance on data preparation or model selection.

Model Guidance

Random Forest: Best for datasets with a clear target column (classification or regression) and many features. Look for datasets with numeric or categorical columns. Results show feature importances (higher = more impact) and classification metrics (precision, recall).
Logistic Regression: Ideal for binary classification datasets (e.g., yes/no target). Check for a target with 2 unique values. Coefficients show feature impact (positive/negative).
PCA + KMeans Clustering: Great for clustering datasets with numeric features. Results show cluster counts and PCA variance (higher % = better explained data).
Sentiment Analysis (DistilBERT): Use datasets with a text column (e.g., reviews). Results show sentiment counts (positive/negative) and scores.
Anomaly Detection: Works for datasets with numeric columns to spot outliers. Results list anomaly records.
TimeSeriesForecasting: Needs a date and value column (e.g., sales over time). Results include forecasts and visualizations.
