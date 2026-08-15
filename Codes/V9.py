# ============================================================
# Heart Attack / Heart Disease Prediction - V9
# Leakage-controlled multi-source evaluation framework
# ============================================================

# If needed:
# pip install pandas numpy scipy scikit-learn imbalanced-learn lightgbm xgboost matplotlib openpyxl shap

import os
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, avoids Tkinter/thread errors
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from scipy import stats

from sklearn.base import clone, BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    GridSearchCV,
    cross_validate
)

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
# 1. Metrics from sklearn.metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    RocCurveDisplay,
)

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from imblearn import FunctionSampler
from imblearn.pipeline import Pipeline as ImbPipeline


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUTPUT_DIR = "V9_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. DATA LOADING
# ============================================================

df1 = pd.read_csv("Heart_Attack_Data_Set.csv")
df2 = pd.read_csv("heart.csv")
df3 = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")

print("Dataset 1:", df1.shape)
print("Dataset 2:", df2.shape)
print("Dataset 3:", df3.shape)

# Rename columns in Dataset 3 to match the other datasets

df3 = df3.rename(
    columns={
        "chest pain type": "cp",
        "resting bp s": "trestbps",
        "cholesterol": "chol",
        "fasting blood sugar": "fbs",
        "resting ecg": "restecg",
        "max heart rate": "thalach",
        "exercise angina": "exang",
        "oldpeak": "oldpeak",
        "ST slope": "slope",
        "target": "target"
    }
)
# ============================================================
# SEMANTIC HARMONIZATION OF IEEE CATEGORICAL CODES
# ============================================================

# Kaggle coding:
# cp:    0=typical, 1=atypical, 2=non-anginal, 3=asymptomatic
# slope: 0=upsloping, 1=flat, 2=downsloping

# IEEE DataPort coding:
# cp:    1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic
# slope: 1=upsloping, 2=flat, 3=downsloping

df3["cp"] = df3["cp"].map({
    1: 0,
    2: 1,
    3: 2,
    4: 3
})

df3["slope"] = df3["slope"].map({
    1: 0,
    2: 1,
    3: 2
})
slope_unmapped_count = int(df3["slope"].isna().sum())

print(
    "IEEE slope values converted to missing "
    "because they were outside the documented coding:",
    slope_unmapped_count
)
# ============================================================
# 2. FEATURE HARMONIZATION
# ============================================================

# Features shared across all three datasets
COMMON_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "target"
]

# ca and thal are deliberately excluded because they are not
# consistently available across all three data sources.

df1_common = df1[COMMON_FEATURES].copy()
df2_common = df2[COMMON_FEATURES].copy()
df3_common = df3[COMMON_FEATURES].copy()

df1_common["source"] = "Kaggle_1"
df2_common["source"] = "Kaggle_2"
df3_common["source"] = "IEEE_DataPort"

final_df = pd.concat(
    [df1_common, df2_common, df3_common],
    ignore_index=True
)

print("Merged dataset shape:", final_df.shape)
print(final_df.head())

# ============================================================
# 3. DATA QUALITY CHECK
# ============================================================

print("\nMissing values:")
print(final_df.isnull().sum())

print("\nDuplicate rows:")
print(final_df.duplicated(subset=COMMON_FEATURES).sum())

print("\nClass distribution:")
print(final_df["target"].value_counts())

print("\nClass percentages:")
print(
    final_df["target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

class_distribution = (
    final_df.groupby("source")["target"]
    .value_counts()
    .unstack(fill_value=0)
)

class_distribution["Total"] = class_distribution.sum(axis=1)

class_distribution["Positive_%"] = (
    class_distribution.get(1, 0)
    / class_distribution["Total"]
    * 100
).round(2)

print(class_distribution)

class_distribution.to_excel(
    os.path.join(OUTPUT_DIR, "class_distribution_by_source.xlsx")
)
# Remove exact duplicate clinical records
before = len(final_df)

final_df = final_df.drop_duplicates(
    subset=COMMON_FEATURES
).reset_index(drop=True)

after = len(final_df)

print("Rows before duplicate removal:", before)
print("Rows after duplicate removal:", after)
print("Duplicates removed:", before - after)

# No global statistical outlier removal is performed.
# However, physiologically implausible zero values in selected continuous
# measurements are treated as missing measurements. This deterministic
# recoding does not use the target variable and therefore does not introduce
# target leakage.

ZERO_AS_MISSING_FEATURES = ["trestbps", "chol"]

zero_quality_rows = []
for col in ZERO_AS_MISSING_FEATURES:
    zero_count = int((final_df[col] == 0).sum())
    zero_quality_rows.append({
        "Feature": col,
        "Zero_count_before_recode": zero_count,
        "Zero_percent_before_recode": round(100.0 * zero_count / len(final_df), 4),
    })
    final_df.loc[final_df[col] == 0, col] = np.nan

zero_quality_df = pd.DataFrame(zero_quality_rows)
zero_quality_df.to_excel(
    os.path.join(OUTPUT_DIR, "implausible_zero_values_recode.xlsx"),
    index=False
)

print("\nPhysiologically implausible zero values recoded as missing:")
print(zero_quality_df)
print("\nMissing values after zero-value recoding:")
print(final_df.isnull().sum())

# Audit missingness created by physiologically implausible zero recoding.
# These indicators are saved only for transparency and are NOT used as model
# predictors, which avoids allowing source-specific missingness patterns to
# become an explicit shortcut for classification.
missingness_audit = (
    final_df.assign(
        trestbps_missing=final_df["trestbps"].isna().astype(int),
        chol_missing=final_df["chol"].isna().astype(int)
    )
    .groupby(["source", "target"])[["trestbps_missing", "chol_missing"]]
    .sum()
    .reset_index()
)
missingness_audit.to_excel(
    os.path.join(OUTPUT_DIR, "missingness_audit_by_source_and_target.xlsx"),
    index=False
)

# Class distribution after exact duplicate removal.
post_dedup_class = final_df["target"].value_counts().sort_index()
post_dedup_class_df = pd.DataFrame({
    "Target": post_dedup_class.index.astype(int),
    "Count": post_dedup_class.values
})
post_dedup_class_df["Percent"] = (
    post_dedup_class_df["Count"] / len(final_df) * 100
).round(2)
post_dedup_class_df.to_excel(
    os.path.join(OUTPUT_DIR, "class_distribution_after_duplicate_removal.xlsx"),
    index=False
)
print("\nClass distribution after duplicate removal:")
print(post_dedup_class_df)

# Descriptive summaries for the valid observed numerical measurements.
# NaN values created from implausible zero measurements are ignored here;
# model imputation is performed later and only inside training folds.
EDA_NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
eda_numeric_summary = (
    final_df.groupby("target")[EDA_NUMERIC_FEATURES]
    .agg(["count", "mean", "std", "median", "min", "max"])
    .round(3)
)
eda_numeric_summary.to_excel(
    os.path.join(OUTPUT_DIR, "eda_numeric_summary_by_target.xlsx")
)

# Figure 1: categorical/discrete predictors by target.
EDA_CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope"]
EDA_CATEGORICAL_TITLES = [
    "Sex",
    "Chest Pain Type",
    "Fasting Blood Sugar",
    "Resting ECG",
    "Exercise-Induced Angina",
    "ST-Segment Slope"
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
for ax, var, title in zip(axes.flatten(), EDA_CATEGORICAL_FEATURES, EDA_CATEGORICAL_TITLES):
    counts = pd.crosstab(final_df[var], final_df["target"]).sort_index()
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(var)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Target")

fig.suptitle("Distribution of Categorical Features by Heart Disease Status", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(
    os.path.join(OUTPUT_DIR, "figure1_categorical_features.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# Figure 2: numerical predictors by target. Implausible zero-coded values have
# already been converted to NaN and are therefore excluded from KDE estimation.
EDA_NUMERIC_TITLES = [
    "Age",
    "Resting Blood Pressure",
    "Cholesterol",
    "Maximum Heart Rate",
    "Oldpeak"
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
axes = axes.flatten()
for ax, var, title in zip(axes, EDA_NUMERIC_FEATURES, EDA_NUMERIC_TITLES):
    observed_min = final_df[var].dropna().min()
    observed_max = final_df[var].dropna().max()
    for target_value in [0, 1]:
        values = final_df.loc[final_df["target"] == target_value, var].dropna()
        values.plot(kind="kde", ax=ax, label=f"Target {target_value}")
    ax.set_xlim(observed_min, observed_max)
    ax.set_title(title)
    ax.set_xlabel(var)
    ax.set_ylabel("Density")
    ax.legend()
axes[-1].axis("off")
fig.suptitle("Distribution of Numerical Clinical Features by Heart Disease Status", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(
    os.path.join(OUTPUT_DIR, "figure2_numerical_features.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# Save the harmonized unique dataset before fold-specific imputation.
final_df.to_excel(
    os.path.join(OUTPUT_DIR, "harmonized_unique_dataset_before_imputation.xlsx"),
    index=False
)

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope"
]

TARGET = "target"

X = final_df[FEATURES].copy()
y = final_df[TARGET].astype(int).copy()

print("X:", X.shape)
print("y:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.40,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True).round(4))

print("\nTest class distribution:")
print(y_test.value_counts(normalize=True).round(4))

# ============================================================
# 4. LEAKAGE-SAFE MEDIAN IMPUTATION + DATA AUGMENTATION
# ============================================================

class DataFrameMedianImputer(BaseEstimator, TransformerMixin):
    """Median-impute selected columns while preserving DataFrame structure.

    Medians are learned only from the data passed to fit(). Because this
    transformer is inside the model pipeline, every CV training fold learns
    its own medians and applies them to its corresponding validation fold.
    No target information is used.
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=FEATURES)
        self.medians_ = X_df[self.columns].median()
        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=FEATURES)
        for col in self.columns:
            X_df[col] = X_df[col].fillna(self.medians_[col])
        return X_df

class DataFrameModeImputer(BaseEstimator, TransformerMixin):
    """Most-frequent imputation while preserving DataFrame structure."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        X_df = (
            X.copy()
            if isinstance(X, pd.DataFrame)
            else pd.DataFrame(X, columns=FEATURES)
        )

        self.modes_ = {}

        for col in self.columns:
            mode_values = X_df[col].mode(dropna=True)

            if len(mode_values) == 0:
                raise ValueError(
                    f"No valid value available to impute column: {col}"
                )

            self.modes_[col] = mode_values.iloc[0]

        return self

    def transform(self, X):
        X_df = (
            X.copy()
            if isinstance(X, pd.DataFrame)
            else pd.DataFrame(X, columns=FEATURES)
        )

        for col in self.columns:
            X_df[col] = X_df[col].fillna(
                self.modes_[col]
            )

        return X_df
median_imputer = DataFrameMedianImputer(
    columns=["trestbps", "chol"]
)

mode_imputer = DataFrameModeImputer(
    columns=["slope"]
)        


def clinical_augmentation(X, y, random_state=42):
    """
    Generate mildly perturbed synthetic samples from training data only.

    Important:
    - No transformation direction depends on target class.
    - Labels are copied only after the feature perturbation process.
    - Intended for use inside CV training folds through FunctionSampler.
    """

    rng = np.random.default_rng(random_state)

    X_df = pd.DataFrame(X, columns=FEATURES).copy()
    y_series = pd.Series(np.asarray(y).ravel()).reset_index(drop=True)

    X_df = X_df.reset_index(drop=True)

    n_original = len(X_df)

    # Create approximately 50% additional training observations
    n_aug = max(1, int(n_original * 0.50))

    sampled_indices = rng.choice(
        np.arange(n_original),
        size=n_aug,
        replace=True
    )

    X_aug = X_df.iloc[sampled_indices].copy()
    y_aug = y_series.iloc[sampled_indices].copy()

    # --------------------------------------------------------
    # Age
    # Small perturbation of +/- 1 year
    # --------------------------------------------------------
    age_noise = rng.choice([-1, 0, 1], size=n_aug)
    X_aug["age"] = X_aug["age"] + age_noise
    X_aug["age"] = X_aug["age"].clip(18, 100)

    # --------------------------------------------------------
    # Cholesterol
    # Target-independent Gaussian perturbation
    # --------------------------------------------------------
    chol_noise = rng.normal(
        loc=0,
        scale=5,
        size=n_aug
    )

    X_aug["chol"] = X_aug["chol"] + chol_noise
    X_aug["chol"] = X_aug["chol"].clip(80, 700)

    # --------------------------------------------------------
    # Resting blood pressure
    # --------------------------------------------------------
    bp_noise = rng.normal(
        loc=0,
        scale=3,
        size=n_aug
    )

    X_aug["trestbps"] = X_aug["trestbps"] + bp_noise
    X_aug["trestbps"] = X_aug["trestbps"].clip(70, 250)

    # --------------------------------------------------------
    # Maximum heart rate
    # --------------------------------------------------------
    hr_noise = rng.normal(
        loc=0,
        scale=3,
        size=n_aug
    )

    X_aug["thalach"] = X_aug["thalach"] + hr_noise
    X_aug["thalach"] = X_aug["thalach"].clip(50, 230)

    # --------------------------------------------------------
    # ST depression
    # --------------------------------------------------------
    oldpeak_noise = rng.normal(
        loc=0,
        scale=0.05,
        size=n_aug
    )

    X_aug["oldpeak"] = X_aug["oldpeak"] + oldpeak_noise
    X_aug["oldpeak"] = X_aug["oldpeak"].clip(0, 10)

    # Categorical variables remain unchanged

    X_resampled = pd.concat(
        [X_df, X_aug],
        ignore_index=True
    )

    y_resampled = pd.concat(
        [y_series, y_aug],
        ignore_index=True
    )

    return X_resampled, y_resampled

# Demonstration only. The actual pipeline repeats these operations separately
# inside every CV training fold.
_demo_imputer = clone(median_imputer)
X_train_imputed_demo = _demo_imputer.fit_transform(X_train, y_train)

X_aug_test, y_aug_test = clinical_augmentation(
    X_train_imputed_demo,
    y_train,
    random_state=RANDOM_STATE
)

print("Before augmentation:", X_train.shape)
print("After training-only median imputation:", X_train_imputed_demo.shape)
print("After augmentation:", X_aug_test.shape)
print("Training-only medians used in demonstration:", _demo_imputer.medians_.to_dict())

print("\nClass distribution before:")
print(y_train.value_counts(normalize=True).round(3))

print("\nClass distribution after:")
print(y_aug_test.value_counts(normalize=True).round(3))

augmenter = FunctionSampler(
    func=clinical_augmentation,
    kw_args={"random_state": RANDOM_STATE},
    validate=False
)

NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope"
]




models = {
    "Logistic Regression":
        LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE
        ),

    "KNN":
        KNeighborsClassifier(),

    "SVM":
        SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

    "Random Forest":
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

    "Ridge Classifier":
        RidgeClassifier(),

    "LightGBM":
        LGBMClassifier(
            random_state=RANDOM_STATE,
            verbose=-1
        ),

    "XGBoost":
        XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1
        )
}

param_grids = {

    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l2"]
    },

    "KNN": {
        "model__n_neighbors": [3, 5, 7, 9, 11],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2]
    },

    "SVM": {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"]
    },

    "Decision Tree": {
        "model__max_depth": [3, 5, 10, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4]
    },

    "Random Forest": {
        "model__n_estimators": [100, 300, 500],
        "model__max_depth": [5, 10, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"]
    },

    "Ridge Classifier": {
        "model__alpha": [0.01, 0.1, 1, 10, 100]
    },

    "LightGBM": {
        "model__n_estimators": [100, 300, 500],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 5, -1],
        "model__num_leaves": [7, 15, 31],
        "model__subsample": [0.8, 1.0]
    },

    "XGBoost": {
        "model__n_estimators": [100, 300, 500],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth": [3, 5, 7],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__gamma": [0, 0.1]
    }
}


def build_pipeline(model):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES
            )
        ],
        remainder="passthrough"
    )

    pipeline = ImbPipeline(
        steps=[
            (
                "numeric_imputation",
                clone(median_imputer)
            ),
            (
                "categorical_imputation",
                clone(mode_imputer)
            ),
            (
                "augmentation",
                augmenter
            ),
            (
                "preprocessing",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    return pipeline
# ============================================================
# 5. NESTED CROSS-VALIDATION
# ============================================================

inner_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

outer_cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=RANDOM_STATE
)

def mean_ci(scores, confidence=0.95):

    scores = np.asarray(scores)

    mean = np.mean(scores)
    sd = np.std(scores, ddof=1)

    sem = stats.sem(scores)

    ci = stats.t.interval(
        confidence,
        len(scores) - 1,
        loc=mean,
        scale=sem
    )

    return {
        "mean": mean,
        "sd": sd,
        "ci_lower": ci[0],
        "ci_upper": ci[1]
    }

nested_results = []
nested_fold_scores = {}

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    pipeline = build_pipeline(model)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    scores = cross_validate(
        estimator=grid,
        X=X_train,
        y=y_train,
        cv=outer_cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        },
        n_jobs=-1,
        return_estimator=False
    )

    nested_fold_scores[model_name] = scores

    row = {"Model": model_name}

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]:

        metric_scores = scores[f"test_{metric}"]

        stat = mean_ci(metric_scores)

        row[f"{metric}_mean"] = stat["mean"]
        row[f"{metric}_sd"] = stat["sd"]
        row[f"{metric}_ci_low"] = stat["ci_lower"]
        row[f"{metric}_ci_high"] = stat["ci_upper"]

    nested_results.append(row)

nested_results_df = pd.DataFrame(nested_results)

nested_results_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "nested_cv_results_with_95CI.xlsx"
    ),
    index=False
)

nested_results_df

# ============================================================
# SAVE NESTED CV FOLD-LEVEL RESULTS
# ============================================================

fold_level_rows = []

for model_name, scores in nested_fold_scores.items():

    n_folds = len(scores["test_roc_auc"])

    for fold_idx in range(n_folds):

        fold_level_rows.append({
            "Model": model_name,
            "Outer_Fold": fold_idx + 1,
            "Accuracy": scores["test_accuracy"][fold_idx],
            "Precision": scores["test_precision"][fold_idx],
            "Recall": scores["test_recall"][fold_idx],
            "F1": scores["test_f1"][fold_idx],
            "ROC_AUC": scores["test_roc_auc"][fold_idx]
        })

nested_fold_level_df = pd.DataFrame(
    fold_level_rows
)

nested_fold_level_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "nested_cv_fold_level_results.xlsx"
    ),
    index=False
)
# ============================================================
# FRIEDMAN TEST - NESTED CV ROC-AUC
# ============================================================

from scipy.stats import friedmanchisquare

model_names = list(nested_fold_scores.keys())

roc_auc_scores = [
    nested_fold_scores[model_name]["test_roc_auc"]
    for model_name in model_names
]

friedman_stat, friedman_p = friedmanchisquare(
    *roc_auc_scores
)

print("\n" + "=" * 70)
print("FRIEDMAN TEST - NESTED CV ROC-AUC")
print("=" * 70)
print(f"Number of models     : {len(model_names)}")
print(f"Number of outer folds: {len(roc_auc_scores[0])}")
print(f"Friedman statistic   : {friedman_stat:.6f}")
print(f"p-value              : {friedman_p:.10f}")

if friedman_p < 0.05:
    print("Result: Significant overall difference among the models.")
else:
    print("Result: No significant overall difference among the models.")


# Save Friedman result
friedman_result_df = pd.DataFrame([
    {
        "Metric": "ROC_AUC",
        "Number_of_models": len(model_names),
        "Number_of_outer_folds": len(roc_auc_scores[0]),
        "Friedman_statistic": friedman_stat,
        "p_value": friedman_p,
        "Significant_at_0.05": friedman_p < 0.05
    }
])

friedman_result_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "friedman_test_results.xlsx"
    ),
    index=False
)
# ============================================================
# 6. REPEATED STRATIFIED K-FOLD
# ============================================================

repeated_cv = RepeatedStratifiedKFold(
    n_splits=10,
    n_repeats=5,
    random_state=RANDOM_STATE
)

repeated_results = []

for model_name, model in models.items():

    print("\nRunning:", model_name)

    pipeline = build_pipeline(model)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    scores = cross_validate(
        estimator=grid,
        X=X_train,
        y=y_train,
        cv=repeated_cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        },
        n_jobs=-1
    )

    row = {"Model": model_name}

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc"
    ]:

        values = scores[f"test_{metric}"]

        stat = mean_ci(values)

        row[f"{metric}_mean"] = stat["mean"]
        row[f"{metric}_sd"] = stat["sd"]
        row[f"{metric}_ci_low"] = stat["ci_lower"]
        row[f"{metric}_ci_high"] = stat["ci_upper"]

    repeated_results.append(row)

repeated_results_df = pd.DataFrame(
    repeated_results
)

repeated_results_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "repeated_kfold_results.xlsx"
    ),
    index=False
)

repeated_results_df

# ============================================================
# 7. FINAL HOLDOUT TEST EVALUATION
# ============================================================

final_results = []
best_estimators = {}

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("Final tuning:", model_name)
    print("=" * 70)

    pipeline = build_pipeline(model)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    best_estimators[model_name] = best_model

    y_pred = best_model.predict(X_test)

    if hasattr(best_model, "predict_proba"):

        y_prob = best_model.predict_proba(
            X_test
        )[:, 1]

        auc = roc_auc_score(
            y_test,
            y_prob
        )

        brier = brier_score_loss(
            y_test,
            y_prob
        )

    elif hasattr(best_model, "decision_function"):

        y_score = best_model.decision_function(
            X_test
        )

        auc = roc_auc_score(
            y_test,
            y_score
        )

        brier = np.nan

    else:
        auc = np.nan
        brier = np.nan

    result = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": auc,
        "Brier_Score": brier,
        "Best_Params": str(grid.best_params_)
    }
# IMPORTANT: save the current model result
    final_results.append(result)
    

# Save progress after each completed model
    pd.DataFrame(final_results).to_excel(
        os.path.join(
            OUTPUT_DIR,
            "final_holdout_test_results_PROGRESS.xlsx"
        ),
        index=False
    )
    print(result)   
    
final_results_df = pd.DataFrame(
    final_results
)

final_results_df = final_results_df.sort_values(
    by="ROC_AUC",
    ascending=False
)

final_results_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "final_holdout_test_results.xlsx"
    ),
    index=False
)

final_results_df

train_test_scores = []

for model_name, model in best_estimators.items():

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_test_scores.append(
        {
            "Model": model_name,

            "Train_Accuracy":
                accuracy_score(
                    y_train,
                    train_pred
                ),

            "Test_Accuracy":
                accuracy_score(
                    y_test,
                    test_pred
                )
        }
    )

train_test_df = pd.DataFrame(
    train_test_scores
)

train_test_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "correct_train_test_accuracy.xlsx"
    ),
    index=False
)

train_test_df

# ============================================================
# 8. STATISTICAL MODEL COMPARISON
# ============================================================

model_names = list(
    nested_fold_scores.keys()
)

auc_matrix = pd.DataFrame(
    {
        model:
            nested_fold_scores[model][
                "test_roc_auc"
            ]
        for model in model_names
    }
)

auc_matrix


from itertools import combinations

pairwise_results = []

for model_a, model_b in combinations(
    model_names,
    2
):

    stat, p = stats.wilcoxon(
        auc_matrix[model_a],
        auc_matrix[model_b],
        zero_method="wilcox"
    )

    pairwise_results.append(
        {
            "Model_A": model_a,
            "Model_B": model_b,
            "Wilcoxon_stat": stat,
            "p_value": p
        }
    )

pairwise_df = pd.DataFrame(
    pairwise_results
)

number_of_tests = len(pairwise_df)

pairwise_df["p_bonferroni"] = np.minimum(
    pairwise_df["p_value"]
    * number_of_tests,
    1.0
)

pairwise_df["significant_0.05"] = (
    pairwise_df["p_bonferroni"] < 0.05
)

pairwise_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "pairwise_wilcoxon_tests.xlsx"
    ),
    index=False
)

pairwise_df

# ============================================================
# 9. CALIBRATION ANALYSIS
# ============================================================

plt.figure(figsize=(8, 7))

for model_name, model in best_estimators.items():

    if not hasattr(model, "predict_proba"):
        continue

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    prob_true, prob_pred = calibration_curve(
        y_test,
        y_prob,
        n_bins=10,
        strategy="quantile"
    )

    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        label=model_name
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration"
)

plt.xlabel("Mean predicted probability")
plt.ylabel("Observed positive proportion")
plt.title("Calibration Curves")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "calibration_curves.png"
    ),
    dpi=300
)
plt.close()


plt.figure(figsize=(8, 7))

ax = plt.gca()

for model_name, model in best_estimators.items():

    try:
        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            name=model_name,
            ax=ax
        )

    except Exception:
        pass

plt.title("ROC Curves on Independent Holdout Test Set")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "roc_curves_all_models.png"
    ),
    dpi=300
)
plt.close()


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

rf_pipeline = best_estimators[
    "Random Forest"
]

rf_model = rf_pipeline.named_steps[
    "model"
]

importances = rf_model.feature_importances_

feature_names = (
    NUMERIC_FEATURES
    +
    CATEGORICAL_FEATURES
)

feature_importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importances
    }
).sort_values(
    "Importance",
    ascending=False
)

feature_importance_df.to_excel(
    os.path.join(
        OUTPUT_DIR,
        "random_forest_feature_importance.xlsx"
    ),
    index=False
)

feature_importance_df


plt.figure(figsize=(9, 6))

plot_df = feature_importance_df.sort_values(
    "Importance",
    ascending=True
)

plt.barh(
    plot_df["Feature"],
    plot_df["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Clinical Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "random_forest_feature_importance.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 11. V9 DATA-QUALITY SUMMARY
# ============================================================

v9_summary = pd.DataFrame([
    {
        "Initial_source_records": before,
        "Unique_records_after_duplicate_removal": after,
        "Duplicates_removed": before - after,
        "Cholesterol_zero_recoded_missing":
            int(
                zero_quality_df.loc[
                    zero_quality_df["Feature"] == "chol",
                    "Zero_count_before_recode"
                ].iloc[0]
            ),
        "RestingBP_zero_recoded_missing":
            int(
                zero_quality_df.loc[
                    zero_quality_df["Feature"] == "trestbps",
                    "Zero_count_before_recode"
                ].iloc[0]
            ),
        "Slope_invalid_code_recoded_missing":
            slope_unmapped_count,
        "Holdout_test_fraction": 0.40,
        "Random_state": RANDOM_STATE
    }
])

v9_summary.to_excel(
    os.path.join(OUTPUT_DIR, "V9_data_quality_summary.xlsx"),
    index=False
)

# ============================================================
# 12. OPTIONAL SHAP ANALYSIS
# ============================================================
#import shap
#
#rf_pipeline = best_estimators["Random Forest"]
#
#X_test_imputed = (
#    rf_pipeline
#    .named_steps["imputation"]
#    .transform(X_test)
#)

#X_test_processed = (
#    rf_pipeline
#    .named_steps["preprocessing"]
#    .transform(X_test_imputed)
#)
#
#rf_model = rf_pipeline.named_steps["model"]
#
#explainer = shap.TreeExplainer(rf_model)
#shap_values = explainer.shap_values(
#    X_test_processed
#)
#
#shap.summary_plot(
#    shap_values,
#    X_test_processed,
#    feature_names=feature_names
#)
