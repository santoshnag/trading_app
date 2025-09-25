"""Machine learning models and training routines for the intraday strategy.

This module defines helper functions to train classification models on
feature matrices and generate probability forecasts.  Two model families
are supported out of the box: logistic regression (a linear classifier
suitable for quick prototyping and interpretable coefficients) and
gradient boosting via XGBoost (more flexible and capable of capturing
non‑linear interactions).

Models are trained using a walk‑forward approach.  The dataset is
partitioned into sequential train and test periods according to the
``TRAIN_WINDOW_MONTHS`` parameter in the global configuration.  For each
window, a fresh model is fitted on the training data and then used to
predict probabilities on the subsequent test period.  This procedure
mimics a realistic live trading scenario where a model is periodically
retrained on the most recent history and evaluated on unseen data.  It
also avoids lookahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, brier_score_loss
# Optional imports for enhanced features
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    ImbPipeline = Pipeline  # type: ignore

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None  # type: ignore

from sklearn.ensemble import StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE
import numpy as np

# Import config with fallback for both package and script execution
try:
    from . import config
except ImportError:
    import config


def _build_model(model_type: str) -> object:
    """Instantiate a model given a type string.

    Parameters
    ----------
    model_type : str
        Either ``"logistic"``, ``"xgboost"``, ``"lightgbm"``, or ``"ensemble"``.
        Raises ``ValueError`` for unsupported types.

    Returns
    -------
    object
        A scikit‑learn compatible estimator implementing ``fit`` and
        ``predict_proba``.
    """
    if model_type == "logistic":
        # Pipeline with StandardScaler and LogisticRegression
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=500,
            )),
        ])
    elif model_type == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed. Install it or choose 'logistic'.")
        # Enhanced pipeline with feature validation and class weighting
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=30,  # Very fast training
                max_depth=2,  # Shallow trees
                learning_rate=0.1,  # Fast learning
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                min_child_weight=3,
                reg_alpha=0.05,
                reg_lambda=1.0,
                scale_pos_weight=2.0,  # Balanced weighting
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=0,
                random_state=42,
            )),
        ])
    elif model_type == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed. Install it or choose 'logistic'.")
        # Optimized hyperparameters for structured data
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=0,
                verbose=-1,
            )),
        ])
    elif model_type == "ensemble":
        # Advanced ensemble with feature selection for 70%+ win rates
        base_estimators = []

        # Base pipeline with feature selection - remove constant features first
        base_pipeline = Pipeline([
            ("variance_filter", VarianceThreshold(threshold=1e-6)),  # Remove constant/low-variance features
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=15)),  # Select top 15 most predictive features
        ])

        if True:  # Always include logistic
            logistic_pipeline = Pipeline([
                ("variance_filter", VarianceThreshold(threshold=1e-6)),
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif, k=15)),
                ("clf", LogisticRegression(
                    penalty="l2",
                    C=0.5,  # More regularization
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=500,
                    random_state=42,
                )),
            ])
            base_estimators.append(("logistic", logistic_pipeline))

        if XGBClassifier is not None:
            xgb_pipeline = Pipeline([
                ("variance_filter", VarianceThreshold(threshold=1e-6)),
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif, k=15)),
                ("clf", XGBClassifier(
                    n_estimators=80,  # Optimized for speed and accuracy
                    max_depth=2,  # Shallower for less overfitting
                    learning_rate=0.05,  # Balanced learning rate
                    subsample=0.8,  # Good regularization
                    colsample_bytree=0.8,
                    gamma=0.2,  # Moderate gamma
                    min_child_weight=8,  # Balanced
                    reg_alpha=0.3,  # Moderate L1 regularization
                    reg_lambda=3.0,  # Moderate L2 regularization
                    scale_pos_weight=3.0,  # Higher weight for minority class
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=0,
                    random_state=42,
                )),
            ])
            base_estimators.append(("xgboost", xgb_pipeline))

        if LGBMClassifier is not None:
            lgb_pipeline = Pipeline([
                ("variance_filter", VarianceThreshold(threshold=1e-6)),
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif, k=15)),
                ("clf", LGBMClassifier(
                    n_estimators=60,  # Optimized for speed
                    max_depth=2,  # Shallower trees
                    learning_rate=0.05,  # Balanced learning rate
                    subsample=0.8,  # Good regularization
                    colsample_bytree=0.8,
                    reg_alpha=0.2,  # Moderate L1 regularization
                    reg_lambda=2.0,  # Moderate L2 regularization
                    min_child_samples=15,  # Balanced splits
                    n_jobs=0,
                    verbose=-1,
                    random_state=42,
                )),
            ])
            base_estimators.append(("lightgbm", lgb_pipeline))

        # Meta-learner: Logistic regression on base model predictions
        meta_learner = LogisticRegression(
            penalty="l2",
            C=0.5,  # More regularization for meta-learner
            class_weight="balanced",
            solver="lbfgs",
            max_iter=500,
            random_state=42,
        )

        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=3,
            stack_method="predict_proba",
            passthrough=False,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: logistic, xgboost, lightgbm, ensemble")


def walk_forward_predict(
    X: DataFrame,
    y: Series,
    timestamps: Series,
    model_type: str | None = None,
    train_window_months: int | None = None,
) -> tuple[Series, Series]:
    """Perform walk‑forward training and prediction.

    The dataset is divided into chronological folds.  For each fold, a model
    is trained on the preceding ``train_window_months`` months of data and
    then used to predict probabilities on the next month of data.  All
    predictions and true labels are concatenated across folds.  This
    function returns two series aligned on timestamps: the predicted
    probability of a positive outcome and the corresponding true label.

    Parameters
    ----------
    X : DataFrame
        Feature matrix with rows indexed by timestamps.
    y : Series
        Target labels aligned with ``X``.
    timestamps : Series
        DatetimeIndex or Series of timestamps corresponding to each row.
    model_type : str, optional
        Type of model to use (``"logistic"`` or ``"xgboost"``).  Defaults to
        ``config.MODEL_TYPE``.
    train_window_months : int, optional
        Size of the rolling training window in months.  Defaults to
        ``config.TRAIN_WINDOW_MONTHS``.

    Returns
    -------
    tuple[Series, Series]
        A tuple of (predicted probabilities, true labels) with the same
        index as the input.
    """
    if model_type is None:
        model_type = config.MODEL_TYPE
    if train_window_months is None:
        train_window_months = config.TRAIN_WINDOW_MONTHS

    # Ensure timestamps are a tz-naive DatetimeIndex (avoid tz-aware comparisons)
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.to_datetime(timestamps)
    else:
        timestamps = pd.DatetimeIndex(timestamps)
    if timestamps.tz is not None:
        # Convert to UTC then drop timezone to compare with naive month boundaries
        timestamps = timestamps.tz_convert("UTC").tz_localize(None)

    # Sort data by timestamps just in case
    order = np.argsort(timestamps.values)
    X = X.iloc[order]
    y = y.iloc[order]
    timestamps = timestamps[order]

    # Convert multi-class labels {-1,0,1} to binary 1 vs rest for training
    y_binary = (y == 1).astype(int)

    preds = pd.Series(index=y.index, dtype=float)
    true_labels = pd.Series(index=y.index, dtype=float)

    # Determine split points by month boundaries
    start_time = timestamps.min()
    end_time = timestamps.max()
    month_starts: list[pd.Timestamp] = []
    current = pd.Timestamp(year=start_time.year, month=start_time.month, day=1)
    while current <= end_time:
        month_starts.append(current)
        current = (current + pd.offsets.MonthBegin()).normalize()
    month_starts.append(end_time + pd.Timedelta(days=1))

    performed_any_fold = False
    # Iterate through months: train on previous train_window_months months, test on next month
    for i in range(train_window_months, len(month_starts) - 1):
        train_start = month_starts[i - train_window_months]
        train_end = month_starts[i]
        test_start = month_starts[i]
        test_end = month_starts[i + 1]
        train_mask = (timestamps >= train_start) & (timestamps < train_end)
        test_mask = (timestamps >= test_start) & (timestamps < test_end)
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        performed_any_fold = True
        X_train = X.loc[train_mask]
        y_train = y_binary.loc[train_mask]
        X_test = X.loc[test_mask]
        y_test = y.loc[test_mask]

        # Skip fold if training set lacks class diversity
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2 or not (0 in unique_classes and 1 in unique_classes):
            # Mark as not performed; continue to next fold
            continue

        model = _build_model(model_type)
        try:
            model.fit(X_train, y_train)
        except ValueError:
            # If fitting fails due to class issues or invalid data, skip this fold
            continue

        performed_any_fold = True
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            classes_ = model.classes_ if hasattr(model, "classes_") else [0, 1]
            try:
                idx_pos = list(classes_).index(1)
            except ValueError:
                idx_pos = np.argmax(classes_)
            prob_pos = proba[:, idx_pos]
        else:
            scores = model.decision_function(X_test)
            prob_pos = 1 / (1 + np.exp(-scores))
        preds.loc[test_mask] = prob_pos
        true_labels.loc[test_mask] = y_test.astype(float)

    # Fallback: Simple train/test split for robust predictions
    if not performed_any_fold:
        print("Using simple train/test split for robust predictions")
        n_samples = len(X)
        train_size = int(0.7 * n_samples)  # 70% for training

        X_train = X.iloc[:train_size]
        y_train = y_binary.iloc[:train_size]
        X_test = X.iloc[train_size:]
        y_test = y.iloc[train_size:]

        print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

        # Ensure we have both classes in training data
        unique_classes = np.unique(y_train)
        if len(unique_classes) >= 2 and (0 in unique_classes and 1 in unique_classes):
            print(f"Training with classes: {unique_classes}")

            # Validate training data for infinite values
            if np.isfinite(X_train.values).all():
                model = _build_model(model_type)
                model.fit(X_train, y_train)

                # Validate test data for infinite values
                if np.isfinite(X_test.values).all():
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_test)
                        classes_ = model.classes_ if hasattr(model, "classes_") else [0, 1]
                        try:
                            idx_pos = list(classes_).index(1)
                        except ValueError:
                            idx_pos = np.argmax(classes_)
                        prob_pos = proba[:, idx_pos]
                    else:
                        scores = model.decision_function(X_test)
                        prob_pos = 1 / (1 + np.exp(-scores))

                    preds.iloc[train_size:] = prob_pos
                    true_labels.iloc[train_size:] = y.iloc[train_size:].astype(float)
                else:
                    print("Warning: Test data contains infinite values")
            else:
                print("Warning: Training data contains infinite values")
        else:
            print(f"Warning: Insufficient class diversity in training data. Classes: {unique_classes}")
            # Fallback: predict neutral (0.5 probability)
            preds.iloc[train_size:] = 0.5
            true_labels.iloc[train_size:] = y.iloc[train_size:].astype(float)
    return preds, true_labels


def tune_probability_threshold(
    preds: Series,
    true_labels: Series,
    candidate_thresholds: list[float] | None = None,
    allow_shorts: bool = False,
) -> float:
    """Choose threshold that maximizes win rate on a validation set.

    This function computes directional trades from probabilities and true labels
    using a threshold and returns the threshold with the highest win rate.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,0.90]

    mask = ~preds.isna() & ~true_labels.isna()
    p = preds.loc[mask]
    y = true_labels.loc[mask]

    best_thr = candidate_thresholds[0]
    best_wr = -1.0
    for thr in candidate_thresholds:
        longs = p >= thr
        shorts = (p <= 1.0 - thr) if allow_shorts else pd.Series(False, index=p.index)
        signals = pd.Series(0, index=p.index, dtype=int)
        signals[longs] = 1
        signals[shorts] = -1
        executed = signals != 0
        if executed.sum() == 0:
            continue
        wins = (signals[executed] == y[executed]).sum()
        wr = wins / executed.sum()
        if wr > best_wr:
            best_wr = wr
            best_thr = thr
    return best_thr


def evaluate_probabilities(preds: Series, true_labels: Series) -> dict[str, float]:
    """Compute basic predictive metrics for probability forecasts.

    Metrics include ROC AUC and Brier score.  The labels are assumed to be
    in {+1, 0, –1}.  For ROC/AUC computation we convert labels to a
    binary indicator of positive vs all other outcomes.

    Parameters
    ----------
    preds : Series
        Predicted probabilities of a positive outcome.
    true_labels : Series
        True labels (–1, 0, +1).

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``"roc_auc"`` and ``"brier"``.  If either
        metric cannot be computed due to lack of positive/negative samples,
        ``nan`` is returned for that metric.
    """
    # Convert true labels to binary: 1 for positive class, 0 otherwise
    mask_valid = ~preds.isna() & ~true_labels.isna()
    y_true_bin = (true_labels.loc[mask_valid] == 1).astype(int)
    y_pred = preds.loc[mask_valid]
    metrics = {}
    # ROC AUC
    try:
        auc = roc_auc_score(y_true_bin, y_pred)
    except Exception:
        auc = np.nan
    metrics["roc_auc"] = auc
    # Brier score
    try:
        brier = brier_score_loss(y_true_bin, y_pred)
    except Exception:
        brier = np.nan
    metrics["brier"] = brier
    return metrics