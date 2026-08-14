# ============================================================
# Heart Attack / Heart Disease Prediction - V7
# Leakage-controlled multi-source evaluation framework
# ============================================================

# If needed:
# pip install pandas numpy scipy scikit-learn imbalanced-learn lightgbm xgboost matplotlib openpyxl shap

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

from scipy import stats

from sklearn.base import clone
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
# 2. Calibration tools from sklearn.calibration
from sklearn.calibration import calibration_curve

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

OUTPUT_DIR = "V7_results"
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

# No global outlier removal is performed.
# Extreme values are retained unless they are clearly impossible.

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
# 4. LEAKAGE-SAFE DATA AUGMENTATION
# ============================================================

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

X_aug_test, y_aug_test = clinical_augmentation(
    X_train,
    y_train,
    random_state=RANDOM_STATE
)

print("Before augmentation:", X_train.shape)
print("After augmentation:", X_aug_test.shape)

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

    pipeline = ImbPipeline(
        steps=[
            ("augmentation", augmenter),
            ("preprocessing", preprocessor),
            ("model", model)
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

        "Accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "Precision":
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),

        "ROC_AUC": auc,

        "Brier_Score": brier,

        "Best_Params":
            str(grid.best_params_)
    }

    final_results.append(result)

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

friedman_stat, friedman_p = stats.friedmanchisquare(
    *[
        auc_matrix[col].values
        for col in auc_matrix.columns
    ]
)

print("Friedman statistic:", friedman_stat)
print("Friedman p-value:", friedman_p)

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

plt.show()

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

plt.show()

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

plt.show()

# ============================================================
# 11. OPTINAL
# ============================================================
import shap
#
rf_pipeline = best_estimators["Random Forest"]
#
X_test_processed = (
    rf_pipeline
    .named_steps["preprocessing"]
    .transform(X_test)
)
#
rf_model = rf_pipeline.named_steps["model"]
#
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(
    X_test_processed
)
#
shap.summary_plot(
    shap_values,
    X_test_processed,
    feature_names=feature_names
)
