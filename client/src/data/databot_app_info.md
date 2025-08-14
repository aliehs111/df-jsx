df.jsx – App Overview & Guide

Purpose: df.jsx is a full-stack web app for uploading, cleaning, exploring, and modeling datasets. It helps non-technical and technical users alike understand their data and prepare it for analysis or machine learning.

Core Features

Dataset Upload

- Upload CSV files up to a reasonable size (e.g., under 100MB for smooth processing).
- Preview the first few rows (e.g., head(10)) in a table view before saving.
- Decide whether to proceed to cleaning or discard the upload using simple buttons.

My Datasets

- View all datasets saved to your account in a paginated table.
- See metadata such as number of rows, columns, data types, missing value percentages, and last updated date.
- Click a dataset to view its details, including options to edit, delete, or run models.

Dataset Detail

- View dataset structure (rows, columns, missing values per column, data types).
- Run info/stats (shape, head, describe, value counts for categorical columns).
- Generate visualizations such as correlation heatmaps, histograms, or box plots for quick insights.
- Save cleaned datasets as new versions for modeling, with version history tracked.

Data Cleaning Options

- Access cleaning tools on the Dataset Detail page via a sidebar or "Clean Data" tab after selecting a dataset.
- Cleaning is non-destructive: Changes are previewed before saving, so you can experiment safely.
- Step-by-step usage:

  1. Go to the Dataset Detail page and locate the cleaning tools in the sidebar or tab.
  2. Select a cleaning action from the dropdown menu (e.g., Handle Missing Values, Scale Features).
  3. Complete the form that appears (e.g., select columns, choose a method like median fill).
  4. Click "Apply Preview" to see changes without saving them.
  5. Review the side-by-side preview table (original data on left, cleaned on right) with changed cells highlighted (e.g., filled values in green).
  6. Check updated stats (e.g., missing counts, scaled ranges) or visualizations (e.g., new histograms) to confirm the effect.
  7. If satisfied, click "Save Cleaned Version" to create a new dataset version; otherwise, click "Reset" to try again.

- Cleaning Options and How to Use Them:

  - Handle Missing Values:
    - Drop: Remove rows or columns with excessive missing data (e.g., set a threshold like 50% missing).
      - Use for: Non-critical columns with mostly missing values.
      - How: Select "Drop" from the dropdown, choose rows/columns, set a threshold, and click "Apply Preview".
      - Tip: Check missing value stats first to avoid losing too much data.
    - Fill with Stats: Replace missing values with the column’s mean, median, or mode.
      - Use for: Numeric columns (mean/median) or categorical columns (mode).
      - How: Select "Fill", choose a stat (e.g., median), pick columns, and click "Apply Preview".
      - Tip: Median is better for skewed numeric data; preview to ensure filled values are reasonable.
    - Custom Fill: Enter a specific value (e.g., 0 or "Unknown") or use forward/backward fill for time-series data.
      - Use for: Specific cases, like setting missing categories to a default.
      - How: Select "Custom Fill", enter the value, choose columns, and preview.
    - KNN Imputation: Fill missing values based on similar rows (advanced, for numeric data).
      - Use for: Accurate fills when you have sufficient data.
      - How: Select "KNN Imputation", set neighbors (e.g., 5), and preview.
      - Tip: Requires clean numeric columns; avoid for small datasets.
  - Scale Features:
    - Normalize: Rescale numeric columns to a [0,1] range (MinMaxScaler).
      - Use for: Features with different scales (e.g., age vs. income) before modeling.
      - How: Select "Normalize", choose numeric columns, and click "Apply Preview".
      - Tip: Preview histograms to confirm the new range.
    - Standardize: Transform numeric columns to mean=0, standard deviation=1 (StandardScaler).
      - Use for: Models like SVM or neural networks that assume normal distributions.
      - How: Select "Standardize", pick columns, and preview the transformed values.
      - Tip: Exclude categorical or target columns to avoid errors.
  - Encode Categorical Variables:
    - One-Hot Encoding: Convert categories to binary columns (e.g., "Red" becomes Red=1 or 0).
      - Use for: Nominal categories (e.g., colors) with few unique values.
      - How: Select "One-Hot", choose categorical columns, and preview new columns.
      - Tip: Avoid for columns with many categories (e.g., 100+ values) to prevent dimension explosion.
    - Label Encoding: Assign integers to categories (e.g., Low=0, Medium=1, High=2).
      - Use for: Ordinal categories with a natural order.
      - How: Select "Label Encoding", pick columns, and preview the integers.
    - Target Encoding: Replace categories with the mean of the target variable (advanced).
      - Use for: High-cardinality categories in predictive models.
      - How: Select "Target Encoding", specify the target column, and preview.
  - Other Tools:
    - Remove Duplicates: Eliminate duplicate rows based on all or specific columns.
      - How: Select "Remove Duplicates", choose columns (or all), and preview.
    - Outlier Detection: Flag or remove outliers using Z-score or IQR methods.
      - How: Select "Outlier Detection", set a threshold, and review flagged rows.
    - Type Conversion: Change column types (e.g., string to numeric or date).
      - How: Select "Type Conversion", pick the new type, and preview.

- Previewing Changes:
  - The preview shows a side-by-side comparison of original and cleaned data.
  - Changed cells are highlighted (e.g., filled values in green) for easy review.
  - Stats (e.g., missing counts, scaled ranges) and visualizations (e.g., heatmaps, histograms) update automatically.
  - Use the preview to check for errors, like over-filling missing values or scaling incorrect columns.
  - Click "Reset" to undo preview changes or "Save Cleaned Version" to keep them.
  - Tip: Preview multiple steps together (e.g., fill then scale) to see combined effects before saving.

Models

- Run pre-trained or built-in models on cleaned datasets.
- Supported models include:
  - Random Forest
  - Logistic Regression
  - PCA + KMeans Clustering
  - Sentiment Analysis (DistilBERT)
  - Anomaly Detection
  - Feature Importance
- View structured results in a clean UI, with options to export predictions or visuals.

Dashboard

- Quick KPI cards (datasets, total rows, columns, last upload).
- Recent Datasets table with quick links.
- Quick Actions for Upload, View All Datasets, Run a Model.

Databot

- Global chatbot that answers questions about the app.
- On the dashboard or models page, Databot acts as a welcome guide, explaining features like data cleaning.
- On dataset or predictor pages, it provides contextual tips based on metadata or model results.
- Ask Databot for step-by-step help, e.g., “How do I fill missing values?” or “What’s the preview feature?”

Typical Workflow

- Upload a CSV file.
- Preview the data and decide whether to save it.
- Clean and preprocess the dataset using the tools above.
- Preview changes thoroughly before saving the cleaned version.
- Run one or more models to analyze the data.
- Review results and download outputs if needed.

Tips

- You must be logged in to use the app.
- Data cleaning steps are saved in version history, so you can revisit or revert them.
- Use Databot for guidance on data preparation or model selection.
- For large datasets, cleaning may take time—monitor progress indicators.
- Always preview changes to avoid data loss; saved versions create copies, not overwrites.

Model Guidance

- Random Forest: Best for datasets with a clear target column (classification or regression) and many features. Results show feature importances (higher = more impact) and classification metrics (precision, recall).
- Logistic Regression: Ideal for binary classification datasets (e.g., yes/no target). Coefficients show feature impact (positive/negative).
- PCA + KMeans Clustering: Great for clustering datasets with numeric features. Results show cluster counts and PCA variance (higher % = better explained data).
- Sentiment Analysis (DistilBERT): Use datasets with a text column (e.g., reviews). Results show sentiment counts (positive/negative) and scores.
- Anomaly Detection: Works for datasets with numeric columns to spot outliers. Results list anomaly records.
- TimeSeriesForecasting: Needs a date and value column (e.g., sales over time). Results include forecasts and visualizations.
