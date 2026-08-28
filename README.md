# uber-ride-analytics

# Uber Ride Cancellations Prediction 🚖

## 📌 Project Overview
This project uses the [Uber Ride Analytics Dashboard dataset](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard) from Kaggle (150,000 ride bookings, 21 columns) to explore ride booking patterns and build a model that predicts whether a customer will **cancel a ride**.

The notebook covers data loading, cleaning, feature engineering, model training and evaluation, and a permutation-based feature importance analysis.

## 🔑 What the notebook does

1. **Data loading** - downloads the dataset via `kagglehub` and loads `ncr_ride_bookings.csv` into a pandas DataFrame.
2. **Initial inspection** - `df.info()`, `df.describe()`, and a null-value count to understand data quality (several columns, e.g. `Avg CTAT`, `Booking Value`, `Driver Ratings`, are only populated for a subset of rides depending on booking outcome).
3. **Feature engineering**
   - Builds a combined `datetime` column from `Date` and `Time`.
   - Defines the target, `target_customer_cancelled`, from the `Booking Status` column and the `Cancelled Rides by Customer` flag.
   - Extracts `hour`, `weekday`, and `is_weekend` from the timestamp.
   - Assembles a numeric feature set (`Avg VTAT`, `Avg CTAT`, `Booking Value`, `Ride Distance`, `Driver Ratings`, `Customer Rating`, plus the time features).
4. **Train/test split** - an 80/20 time-based split (sorted by `datetime`), used for the main pipeline models. A second, stratified random 80/20 split is used later for a numeric-features-only comparison.
5. **Modelling**
   - A `ColumnTransformer` + `Pipeline` handles median imputation for numeric features and most-frequent imputation + one-hot encoding for categoricals.
   - **Logistic Regression** (`class_weight='balanced'`) as a baseline.
   - **Random Forest** (400 trees, `class_weight='balanced_subsample'`) as the main model.
   - A separate pass trains Logistic Regression and Random Forest again using only the numeric columns of the full dataset, evaluated with a stratified split.
6. **Threshold tuning** - a helper function scans the precision-recall curve to pick the probability threshold that maximises F1 for the Random Forest model.
7. **Evaluation** - ROC-AUC, PR-AUC (average precision), classification report, confusion matrix, ROC curve, and precision-recall curve.
8. **Feature importance** - permutation importance computed on the held-out test set for the Random Forest pipeline, plotted as a horizontal bar chart of the top 15 features.

## 📊 Key Insights
- Autos had the highest cancellation rate among vehicle types; premium rides cancelled less often.
- Cancellations peak during evenings and on weekends.
- Certain pickup locations show disproportionately high cancellation rates.
- Lower driver ratings correlate with higher cancellation rates.

## 🎯 Suggested actions (from the analysis)
- Incentivise higher-rated drivers.
- Improve pickup coordination at hotspot locations.
- Increase driver supply during peak cancellation hours.

## 🛠 Tech Stack
- **Python** - pandas, NumPy, Matplotlib, Seaborn
- **scikit-learn** - `ColumnTransformer`, `Pipeline`, `LogisticRegression`, `RandomForestClassifier`, `permutation_importance`
- **kagglehub** - dataset download/caching

## 📁 Data
Dataset: `ncr_ride_bookings.csv` (150,000 rows × 21 columns), downloaded automatically via `kagglehub.dataset_download("yashdevladdha/uber-ride-analytics-dashboard")`. No manual download is needed to re-run the notebook, provided Kaggle credentials are configured locally.

## ⚠️ Notes / possible improvements
- A `leaky_features` list (columns like `Booking Value`, `Ride Distance`, ratings, and post-booking status flags) is defined in the notebook but isn't actually excluded from the feature set used in the second modelling pass - worth double-checking that the reported metrics aren't inflated by information that wouldn't be available at prediction time (i.e. before a ride is completed or cancelled).
- The README text embedded in the original notebook mentions XGBoost as part of the tech stack, but no XGBoost code currently appears in the notebook - that would be a natural next model to add for comparison against the Random Forest baseline.
- Next step suggested in the notebook: deploy the model as an API for real-time cancellation prediction and integrate it into a ride-matching system for proactive driver allocation.
