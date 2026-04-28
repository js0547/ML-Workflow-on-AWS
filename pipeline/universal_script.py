"""
Universal Training Script for the Tabular ML Platform.

This script is strictly command-line driven via argparse.
It executes the exact model and configuration selected by the user in the UI.
No Auto-ML logic -- the user's choice is final.

Usage:
    python universal_script.py \
        --model_type logistic_regression \
        --task_type classification \
        --target_column target \
        --dropped_columns "col1,col2" \
        --train /opt/ml/input/data/train \
        --model_dir /opt/ml/model \
        --output_dir /opt/ml/output/data
"""

import argparse
import json
import logging
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Try to import plotting libraries -- they may not be available or compatible
# in all SageMaker container environments. Training will still work without them.
HAS_PLOTTING = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except (ImportError, RuntimeError) as e:
    logging.warning("Plotting libraries unavailable: %s. EDA plots will be skipped.", e)

# Suppress warnings that clutter SageMaker logs
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Argument Parsing
# ------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments passed by SageMaker or the user."""
    parser = argparse.ArgumentParser(
        description="Train a user-selected ML model on tabular data."
    )

    # User selections (passed as SageMaker hyperparameters)
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=[
            "logistic_regression",
            "linear_regression",
            "xgboost",
            "lightgbm",
        ],
        help="ML model to train. Must be one of the four supported types.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        required=True,
        choices=["classification", "regression"],
        help="Task type: classification or regression.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        required=True,
        help="Name of the target column in the CSV.",
    )
    parser.add_argument(
        "--dropped_columns",
        type=str,
        default="",
        help="Comma-separated list of columns to drop. Empty string means drop none.",
    )

    # SageMaker environment paths
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
        help="Path to the training data directory.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
        help="Path to save EDA plots and evaluation metrics.",
    )

    args, _ = parser.parse_known_args()
    return args


# ------------------------------------------------------------------
# Data Loading
# ------------------------------------------------------------------

def load_data(train_dir):
    """Load CSV data from the SageMaker training channel directory."""
    csv_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    if not csv_files:
        logger.error("No CSV files found in %s", train_dir)
        sys.exit(1)

    csv_path = os.path.join(train_dir, csv_files[0])
    logger.info("Loading data from: %s", csv_path)

    df = pd.read_csv(csv_path)
    logger.info("Dataset shape: %s rows, %s columns", df.shape[0], df.shape[1])
    logger.info("Columns: %s", list(df.columns))

    return df


# ------------------------------------------------------------------
# EDA: Exploratory Data Analysis
# ------------------------------------------------------------------

def generate_eda_plots(df, output_dir):
    """Generate and save EDA plots: correlation heatmap and missing value matrix."""
    os.makedirs(output_dir, exist_ok=True)

    # Always generate the JSON data summary (no plotting library needed)
    generate_data_summary(df, output_dir)

    if not HAS_PLOTTING:
        logger.warning("Skipping EDA plots (matplotlib/seaborn not available).")
        return

    logger.info("Generating EDA plots...")

    # -- Correlation Heatmap --
    generate_correlation_heatmap(df, output_dir)

    # -- Missing Value Matrix --
    generate_missing_value_matrix(df, output_dir)

    logger.info("EDA plots saved to %s", output_dir)


def generate_correlation_heatmap(df, output_dir):
    """Create a correlation heatmap of all numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        logger.warning("Not enough numeric columns for correlation heatmap.")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = numeric_df.corr()

    sns.heatmap(
        correlation_matrix,
        annot=True if numeric_df.shape[1] <= 15 else False,
        fmt=".2f" if numeric_df.shape[1] <= 15 else "",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")
    plt.tight_layout()

    filepath = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation heatmap: %s", filepath)


def generate_missing_value_matrix(df, output_dir):
    """Create a missing value matrix showing the pattern of missing data."""
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    missing_data = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_counts.values,
        "missing_percent": missing_pct.values,
    })

    # Save missing data summary as JSON
    missing_json_path = os.path.join(output_dir, "missing_values.json")
    missing_data.to_json(missing_json_path, orient="records", indent=2)

    # Create a visual matrix
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: heatmap of missing values (white = missing, dark = present)
    ax1 = axes[0]
    missing_matrix = df.isnull().astype(int)
    # Show a sample if dataset is large
    sample_size = min(100, len(missing_matrix))
    sample_matrix = missing_matrix.head(sample_size)

    sns.heatmap(
        sample_matrix,
        cbar=False,
        yticklabels=False,
        cmap=["#2ecc71", "#e74c3c"],
        ax=ax1,
    )
    ax1.set_title(
        "Missing Value Pattern (first {} rows)".format(sample_size),
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xlabel("Columns")

    # Right: bar chart of missing percentages
    ax2 = axes[1]
    cols_with_missing = missing_data[missing_data["missing_count"] > 0].sort_values(
        "missing_percent", ascending=True
    )

    if len(cols_with_missing) > 0:
        ax2.barh(
            cols_with_missing["column"],
            cols_with_missing["missing_percent"],
            color="#e74c3c",
            edgecolor="#c0392b",
        )
        ax2.set_xlabel("Missing (%)")
        ax2.set_title("Missing Values by Column", fontsize=14, fontweight="bold")
        for i, (pct, count) in enumerate(
            zip(cols_with_missing["missing_percent"], cols_with_missing["missing_count"])
        ):
            ax2.text(pct + 0.5, i, "{:.1f}% ({})".format(pct, int(count)), va="center")
    else:
        ax2.text(0.5, 0.5, "No Missing Values", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=16, fontweight="bold",
                 color="#2ecc71")
        ax2.set_title("Missing Values by Column", fontsize=14, fontweight="bold")

    plt.tight_layout()

    filepath = os.path.join(output_dir, "missing_value_matrix.png")
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved missing value matrix: %s", filepath)


def generate_data_summary(df, output_dir):
    """Generate a JSON summary of the dataset."""
    summary = {
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
        "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns),
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # Basic statistics for numeric columns
    if len(summary["numeric_columns"]) > 0:
        desc = df[summary["numeric_columns"]].describe()
        summary["numeric_stats"] = desc.to_dict()

    filepath = os.path.join(output_dir, "data_summary.json")
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved data summary: %s", filepath)


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------

def preprocess_data(df, target_column, dropped_columns, task_type):
    """
    Preprocess the dataset:
    1. Drop user-specified columns
    2. Separate features and target
    3. Handle missing values (median for numeric, mode for categorical)
    4. Encode categorical features
    5. Train/test split (80/20)
    """
    logger.info("Starting preprocessing...")

    # Step 1: Drop user-specified columns
    if dropped_columns:
        cols_to_drop = [c.strip() for c in dropped_columns.split(",") if c.strip()]
        existing_cols = [c for c in cols_to_drop if c in df.columns]
        if existing_cols:
            df = df.drop(columns=existing_cols)
            logger.info("Dropped columns: %s", existing_cols)
        else:
            logger.warning("None of the specified columns found: %s", cols_to_drop)

    # Step 2: Validate target column
    if target_column not in df.columns:
        logger.error("Target column '%s' not found in dataset.", target_column)
        logger.error("Available columns: %s", list(df.columns))
        sys.exit(1)

    y = df[target_column]
    X = df.drop(columns=[target_column])

    logger.info("Target column: %s", target_column)
    logger.info("Feature columns: %s", list(X.columns))

    # Step 3: Handle missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    for col in numeric_cols:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            logger.info("Imputed %s with median: %s", col, median_val)

    for col in categorical_cols:
        if X[col].isnull().any():
            mode_val = X[col].mode()[0]
            X[col] = X[col].fillna(mode_val)
            logger.info("Imputed %s with mode: %s", col, mode_val)

    # Handle missing values in target
    if y.isnull().any():
        if task_type == "classification":
            y = y.fillna(y.mode()[0])
        else:
            y = y.fillna(y.median())
        logger.info("Imputed missing values in target column")

    # Step 4: Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        logger.info("Encoded categorical column: %s (%d classes)", col, len(le.classes_))

    # Encode target if classification and target is categorical
    target_encoder = None
    if task_type == "classification" and y.dtype == "object":
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y), name=target_column)
        logger.info("Encoded target column: %d classes", len(target_encoder.classes_))

    # Step 5: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info("Train set: %d rows, Test set: %d rows", len(X_train), len(X_test))

    return X_train, X_test, y_train, y_test, label_encoders, target_encoder


# ------------------------------------------------------------------
# Model Initialization
# Strictly uses if/elif to initialize the user's chosen model.
# No auto-selection or override logic.
# ------------------------------------------------------------------

def initialize_model(model_type, task_type):
    """
    Initialize the model the user requested.
    Uses a conditional structure to select the exact model.
    If the user chooses Logistic Regression, that is what runs --
    even if XGBoost might perform better.
    """
    logger.info("Initializing model: %s (task: %s)", model_type, task_type)

    if model_type == "logistic_regression":
        if task_type != "classification":
            logger.error("Logistic Regression is only for classification tasks.")
            sys.exit(1)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)

    elif model_type == "linear_regression":
        if task_type != "regression":
            logger.error("Linear Regression is only for regression tasks.")
            sys.exit(1)
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

    elif model_type == "xgboost":
        if task_type == "classification":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_estimators=100,
                verbosity=0,
            )
        else:
            from xgboost import XGBRegressor
            model = XGBRegressor(
                random_state=42,
                n_estimators=100,
                verbosity=0,
            )

    elif model_type == "lightgbm":
        if task_type == "classification":
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(
                verbose=-1,
                random_state=42,
                n_estimators=100,
            )
        else:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(
                verbose=-1,
                random_state=42,
                n_estimators=100,
            )

    else:
        logger.error("Unsupported model type: %s", model_type)
        sys.exit(1)

    logger.info("Model initialized: %s", type(model).__name__)
    return model


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_model(model, X_train, y_train):
    """Train the model on the training data."""
    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, task_type, output_dir):
    """
    Evaluate the model and save metrics as evaluation.json.
    Classification: accuracy, precision, recall, f1
    Regression: rmse, mae, r2
    """
    logger.info("Evaluating model...")
    os.makedirs(output_dir, exist_ok=True)

    predictions = model.predict(X_test)

    if task_type == "classification":
        # Determine averaging strategy based on number of classes
        n_classes = len(np.unique(y_test))
        average = "binary" if n_classes == 2 else "weighted"

        metrics = {
            "task_type": "classification",
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, average=average, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, average=average, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, average=average, zero_division=0)),
            "num_classes": n_classes,
            "test_samples": len(y_test),
        }
    else:
        metrics = {
            "task_type": "regression",
            "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "mae": float(mean_absolute_error(y_test, predictions)),
            "r2_score": float(r2_score(y_test, predictions)),
            "test_samples": len(y_test),
        }

    # Save metrics as JSON (used by SageMaker Model Registry)
    metrics_path = os.path.join(output_dir, "evaluation.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation metrics:")
    for key, value in metrics.items():
        logger.info("  %s: %s", key, value)
    logger.info("Metrics saved to: %s", metrics_path)

    return metrics


# ------------------------------------------------------------------
# Model Saving
# ------------------------------------------------------------------

def save_model(model, model_dir, label_encoders, target_encoder, args):
    """Save the trained model and metadata to the model directory."""
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)
    logger.info("Model saved to: %s", model_path)

    # Save encoders for inference use
    encoders_path = os.path.join(model_dir, "encoders.joblib")
    joblib.dump({
        "label_encoders": label_encoders,
        "target_encoder": target_encoder,
    }, encoders_path)
    logger.info("Encoders saved to: %s", encoders_path)

    # Save model metadata
    metadata = {
        "model_type": args.model_type,
        "task_type": args.task_type,
        "target_column": args.target_column,
        "dropped_columns": args.dropped_columns,
    }
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to: %s", metadata_path)


# ------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------

def main():
    """Main entry point for the training script."""
    logger.info("=" * 60)
    logger.info("Universal Training Script -- Starting")
    logger.info("=" * 60)

    # Parse arguments
    args = parse_args()
    logger.info("Configuration:")
    logger.info("  Model Type: %s", args.model_type)
    logger.info("  Task Type: %s", args.task_type)
    logger.info("  Target Column: %s", args.target_column)
    logger.info("  Dropped Columns: %s", args.dropped_columns if args.dropped_columns else "(none)")
    logger.info("  Train Dir: %s", args.train)
    logger.info("  Model Dir: %s", args.model_dir)
    logger.info("  Output Dir: %s", args.output_dir)

    # Load data
    df = load_data(args.train)

    # Generate EDA plots (before preprocessing, on raw data)
    generate_eda_plots(df, args.output_dir)

    # Preprocess
    X_train, X_test, y_train, y_test, label_encoders, target_encoder = preprocess_data(
        df, args.target_column, args.dropped_columns, args.task_type
    )

    # Initialize model (user's exact choice)
    model = initialize_model(args.model_type, args.task_type)

    # Train
    model = train_model(model, X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, args.task_type, args.output_dir)

    # Save model and artifacts
    save_model(model, args.model_dir, label_encoders, target_encoder, args)

    logger.info("=" * 60)
    logger.info("Universal Training Script -- Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
