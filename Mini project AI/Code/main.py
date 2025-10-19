import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, RocCurveDisplay
)
import matplotlib.pyplot as plt


# ---- Helpers ----
POSSIBLE_TARGETS = ["Class", "class", "is_fraud", "fraud", "Fraud", "target", "TARGET", "label"]

def detect_target(df: pd.DataFrame, user_target: str | None) -> str:
    if user_target and user_target in df.columns:
        return user_target
    for c in POSSIBLE_TARGETS:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a target column. Pass --target <col>. "
        f"Available columns: {list(df.columns)[:20]} ..."
    )

def build_preprocessor(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    # Numeric vs categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_tf = Pipeline(steps=[
        ("scaler", StandardScaler(with_mean=False if len(cat_cols)>0 else True))  # keep CSR compatibility
    ])
    cat_tf = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    if cat_cols:
        pre = ColumnTransformer(
            transformers=[
                ("num", num_tf, num_cols),
                ("cat", cat_tf, cat_cols),
            ],
            remainder="drop"
        )
    else:
        pre = ColumnTransformer(
            transformers=[("num", num_tf, num_cols)],
            remainder="drop"
        )
    return pre


def evaluate(model_name, y_true, y_pred, y_proba):
    return {
        "Model": model_name,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan,
    }


def plot_rocs(rocs, save_path="roc_curves.png"):
    plt.figure()
    for name, disp in rocs:
        disp.plot(ax=plt.gca())
    plt.title("ROC Curves")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"[Saved] {save_path}")


# ---- Main ----
def main(args):
    # Load data
    df = pd.read_csv(args.data)

    # Drop useless id column if present
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    target_col = detect_target(df, args.target)
    print(f"Using target column: {target_col}")

    # Basic cleanup: drop rows with all-NA in features
    df = df.dropna(how="all")

    # Binarize target if it's not already 0/1
    y_raw = df[target_col]
    if y_raw.dtype == "bool":
        y = y_raw.astype(int)
    elif y_raw.dtype.kind in "iu":
        y = y_raw.clip(0,1)  # if labels are 0/1 already, fine
    else:
        # Try to map common strings
        y = y_raw.astype(str).str.lower().map({"fraud":1, "true":1, "yes":1, "1":1}).fillna(0).astype(int)

    X = df.drop(columns=[target_col])

    # Preprocess
    pre = build_preprocessor(df, target_col)

    # Train/Val split (stratified because of imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Common pipeline head: Preprocess -> SMOTE (on training only via imblearn pipeline)
    # Note: SMOTE works after preprocessing since all outputs are numeric.
    base_steps = [
        ("preprocess", pre),
        ("smote", SMOTE(random_state=42)),
        ("model", LogisticRegression())  # placeholder; will be replaced
    ]

    results = []
    rocs = []

    # 1) Logistic Regression (baseline)
    logreg = ImbPipeline(steps=[
        *base_steps[:-1],
        ("model", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_proba = getattr(logreg, "predict_proba")(X_test)[:,1]
    results.append(evaluate("Logistic Regression", y_test, y_pred, y_proba))
    rocs.append(("Logistic Regression", RocCurveDisplay.from_predictions(y_test, y_proba)))

    # 2) Decision Tree
    dt = ImbPipeline(steps=[
        *base_steps[:-1],
        ("model", DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ))
    ])
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    y_proba = getattr(dt, "predict_proba")(X_test)[:,1]
    results.append(evaluate("Decision Tree", y_test, y_pred, y_proba))
    rocs.append(("Decision Tree", RocCurveDisplay.from_predictions(y_test, y_proba)))

    # 3) MLP (ANN)
    mlp = ImbPipeline(steps=[
        *base_steps[:-1],
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=80,
            early_stopping=True,
            random_state=42
        ))
    ])
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    # MLP might not have predict_proba if it fails to converge; handle safely
    y_proba = mlp.predict_proba(X_test)[:,1] if hasattr(mlp, "predict_proba") else None
    results.append(evaluate("MLP (ANN)", y_test, y_pred, y_proba))
    if y_proba is not None:
        rocs.append(("MLP (ANN)", RocCurveDisplay.from_predictions(y_test, y_proba)))

    # Results table
    res_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
    print("\n=== Metrics (higher is better) ===")
    print(res_df.to_string(index=False))

    # Optional: Detailed report for best model
    best_name = res_df.iloc[0]["Model"]
    best_pipe = {"Logistic Regression": logreg, "Decision Tree": dt, "MLP (ANN)": mlp}[best_name]
    best_pred = best_pipe.predict(X_test)
    print(f"\n=== Detailed report: {best_name} ===")
    print(classification_report(y_test, best_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, best_pred))
    
    from sklearn.metrics import ConfusionMatrixDisplay

    # Save confusion matrix as PNG
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, best_pred, ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.savefig("confusion_matrix.png", bbox_inches="tight", dpi=150)
    print("[Saved] confusion_matrix.png")

    # Save ROC curves
    if rocs:
        plot_rocs(rocs, save_path="roc_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection: model comparison")
    parser.add_argument("--data", type=str, default="data/creditcard.csv",
                        help="Path to CSV (place one Kaggle dataset here).")
    parser.add_argument("--target", type=str, default=None,
                        help="Target column (optional). If omitted, will try to auto-detect.")
    args = parser.parse_args()
    main(args)
