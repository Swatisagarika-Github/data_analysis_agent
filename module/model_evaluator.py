
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Regression algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso


# --------------------------------------------------
# 1. PREPROCESSING
# --------------------------------------------------

def create_preprocessor(X):

    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns)
        ],
        remainder="drop"
    )

    return preprocessor


# --------------------------------------------------
# 2. DETECT TASK TYPE
# --------------------------------------------------

def detect_task_type(y):

    if (
        y.dtype == "object"
        or str(y.dtype) == "category"
        or y.dtype == "bool"
    ):
        return "classification"

    unique_values = y.nunique()
    total_values = len(y)

    # Binary target
    if unique_values == 2:
        return "classification"

    # Small number of integer classes
    if (
        pd.api.types.is_integer_dtype(y)
        and unique_values <= 10
        and unique_values / max(total_values, 1) <= 0.05
    ):
        return "classification"

    return "regression"


# --------------------------------------------------
# 3. CLASSIFICATION MODELS
# --------------------------------------------------

def get_classification_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "SVM": SVC(
            kernel="rbf",
            random_state=42
        ),

        "Naive Bayes": GaussianNB()
    }


# --------------------------------------------------
# 4. REGRESSION MODELS
# --------------------------------------------------

def get_regression_models():

    return {
        "Linear Regression": LinearRegression(),

        "Ridge Regression": Ridge(),

        "Lasso Regression": Lasso(),

        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),

        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=42
        )
    }


# --------------------------------------------------
# 5. EVALUATE CLASSIFICATION MODELS
# --------------------------------------------------

def evaluate_classification(
    X_train,
    X_test,
    y_train,
    y_test,
    models,
    preprocessor
):

    results = []
    trained_models = {}

    for model_name, model in models.items():

        try:

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            accuracy = accuracy_score(
                y_test, predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            results.append({
                "algorithm": model_name,
                "accuracy": round(accuracy * 100, 2),
                "precision": round(precision * 100, 2),
                "recall": round(recall * 100, 2),
                "f1_score": round(f1 * 100, 2),
                "status": "Success"
            })

            trained_models[model_name] = pipeline

        except Exception as error:

            results.append({
                "algorithm": model_name,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "status": f"Failed: {str(error)[:150]}"
            })

    return results, trained_models


# --------------------------------------------------
# 6. EVALUATE REGRESSION MODELS
# --------------------------------------------------

def evaluate_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    models,
    preprocessor
):

    results = []
    trained_models = {}

    for model_name, model in models.items():

        try:

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)

            predictions = pipeline.predict(X_test)

            r2 = r2_score(
                y_test, predictions
            )

            mae = mean_absolute_error(
                y_test, predictions
            )

            rmse = np.sqrt(
                mean_squared_error(
                    y_test, predictions
                )
            )

            results.append({
                "algorithm": model_name,
                "r2_score": round(r2 * 100, 2),
                "mae": round(float(mae), 4),
                "rmse": round(float(rmse), 4),
                "status": "Success"
            })

            trained_models[model_name] = pipeline

        except Exception as error:

            results.append({
                "algorithm": model_name,
                "r2_score": None,
                "mae": None,
                "rmse": None,
                "status": f"Failed: {str(error)[:150]}"
            })

    return results, trained_models


# --------------------------------------------------
# 7. MAIN MODEL EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_models(
    df,
    target_column=None,
    task_type=None,
    test_size=0.2,
    random_state=42
):

    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    if len(df) < 10:
        raise ValueError(
            "At least 10 rows are required for model evaluation."
        )

    if target_column is None:
        raise ValueError(
            "Please select a target column before model training."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size must be between 0 and 1."
        )

    data = df.copy()

    # Remove rows with missing target values
    data = data.dropna(subset=[target_column])

    if len(data) < 10:
        raise ValueError(
            "Not enough rows remain after removing missing target values."
        )

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Remove columns that cannot be used as model features
    X = X.drop(
        columns=[
            col for col in X.columns
            if X[col].nunique(dropna=False) <= 1
        ]
    )

    # Remove unsupported datetime features for now
    datetime_columns = X.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    X = X.drop(columns=datetime_columns)

    if X.empty:
        raise ValueError(
            "No usable feature columns found."
        )

    # Remove infinite numeric values
    X = X.replace([np.inf, -np.inf], np.nan)

    if y.nunique() < 2:
        raise ValueError(
            "Target column must contain at least two different values."
        )

    if task_type is None or task_type == "auto":
        task_type = detect_task_type(y)

    if task_type not in ["classification", "regression"]:
        raise ValueError(
            "Task type must be classification or regression."
        )

    if task_type == "regression":
        y = pd.to_numeric(y, errors="coerce")
        valid_rows = y.notna()

        X = X.loc[valid_rows]
        y = y.loc[valid_rows]

    if len(X) < 10:
        raise ValueError(
            "Not enough valid rows for model training."
        )

    if task_type == "classification":

        if y.nunique() < 2:
            raise ValueError(
                "Classification requires at least two target classes."
            )

        class_counts = y.value_counts()

        # Use stratification only when each class can
        # appear in both training and testing datasets.
        number_of_test_rows = int(np.ceil(len(y) * test_size))
        number_of_train_rows = len(y) - number_of_test_rows

        if (
                class_counts.min() >= 2
                and number_of_test_rows >= y.nunique()
                and number_of_train_rows >= y.nunique()
        ):
            stratify = y
        else:
            stratify = None

    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    preprocessor = create_preprocessor(X_train)

    # Select suitable models
    if task_type == "classification":

        models = get_classification_models()

        results, trained_models = evaluate_classification(
            X_train,
            X_test,
            y_train,
            y_test,
            models,
            preprocessor
        )

        metric_name = "accuracy"

        successful_results = [
            result for result in results
            if result["status"] == "Success"
        ]

        best_model = (
            max(
                successful_results,
                key=lambda item: item["accuracy"]
            )
            if successful_results else None
        )

    else:

        models = get_regression_models()

        results, trained_models = evaluate_regression(
            X_train,
            X_test,
            y_train,
            y_test,
            models,
            preprocessor
        )

        metric_name = "r2_score"

        successful_results = [
            result for result in results
            if result["status"] == "Success"
        ]

        best_model = (
            max(
                successful_results,
                key=lambda item: item["r2_score"]
            )
            if successful_results else None
        )

    if not successful_results:
        raise RuntimeError(
            "All machine learning algorithms failed."
        )

    # Sort the results
    results = sorted(
        results,
        key=lambda item: (
            item[metric_name]
            if item[metric_name] is not None
            else float("-inf")
        ),
        reverse=True
    )

    best_algorithm = best_model["algorithm"]

    return {
        "task_type": task_type,
        "target_column": target_column,
        "total_rows": len(X),
        "training_rows": len(X_train),
        "testing_rows": len(X_test),
        "metric": metric_name,
        "results": results,
        "best_algorithm": best_algorithm,
        "best_performance": best_model,
        "trained_models": trained_models,
        "training_performed": True
    }


# --------------------------------------------------
# 8. SIMPLE TEST FUNCTION
# --------------------------------------------------

if __name__ == "__main__":

    print("Model Evaluator Module Loaded Successfully.")

    print("\nAvailable Classification Algorithms:")

    for name in get_classification_models():
        print("-", name)

    print("\nAvailable Regression Algorithms:")

    for name in get_regression_models():
        print("-", name)
