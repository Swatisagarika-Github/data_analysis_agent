
import re
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_KEYWORDS = [
    "target", "label", "outcome", "result", "prediction",
    "class", "status", "purchased", "purchase", "churn",
    "survived", "placed", "placement", "approved", "default",
    "salary", "price", "revenue", "sales", "profit",
    "score", "marks", "rating", "disease", "diagnosis",
    "pass", "failed", "fraud", "risk", "category",
    "response", "success", "failure", "grade", "demand"
]

ID_KEYWORDS = [
    "id", "uuid", "index", "serial", "record",
    "customer_id", "user_id", "employee_id",
    "product_id", "transaction_id"
]

DATE_KEYWORDS = [
    "date", "time", "timestamp", "datetime"
]

SUPPORTED_TASKS = [
    "regression",
    "classification",
    "clustering"
]

MIN_ROWS_FOR_ML = 10


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_date_column(series, column_name):
    """
    Detect date/time columns using their names and data types.
    """

    name = str(column_name).lower()

    if any(word in name for word in DATE_KEYWORDS):
        return True

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    return False


def is_id_column(column_name, series):
    """
    Identify likely identifier columns.
    """

    name = str(column_name).lower().strip()

    # Match names such as ID, customer_id, user_id
    if name == "id" or name.endswith("_id"):
        return True

    if any(word in name for word in ID_KEYWORDS):
        return True

    clean_series = series.dropna()

    if clean_series.empty:
        return False

    unique_ratio = clean_series.nunique() / len(clean_series)

    # A column where almost every value is unique
    # is likely an identifier.
    if unique_ratio >= 0.98:
        return True

    return False


def get_column_type(series):
    """
    Identify the type of a dataset column.
    """

    if pd.api.types.is_bool_dtype(series):
        return "categorical"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if is_date_column(series, ""):
        return "datetime"

    return "categorical"


def get_task_type(series, unique_count, row_count):
    """
    Estimate the ML task associated with a possible target.

    Numeric targets with very few distinct values may be
    classification targets. Other numeric targets are
    treated as possible regression targets.
    """

    if pd.api.types.is_bool_dtype(series):
        return "classification"

    if pd.api.types.is_numeric_dtype(series):

        # Binary numeric target
        if unique_count == 2:
            return "classification"

        # Low-cardinality numeric target
        classification_limit = min(
            20,
            max(2, int(row_count * 0.05))
        )

        if unique_count <= classification_limit:
            return "classification"

        return "regression"

    return "classification"


def calculate_class_distribution(series):
    """
    Calculate the distribution of classes in a classification target.
    """

    counts = series.value_counts(dropna=True)

    if counts.empty:
        return {}

    total = counts.sum()

    distribution = {}

    for label, count in counts.items():
        distribution[str(label)] = round(
            (count / total) * 100, 2
        )

    return distribution


# ============================================================
# TARGET COLUMN DETECTION
# ============================================================

def detect_target_candidates(df):
    """
    Detect and rank possible target columns.

    Ranking considers:
    1. Target-related column names
    2. Data type
    3. Unique values
    4. Missing values
    5. Cardinality
    6. Class distribution
    7. Dataset characteristics

    Returns ranked target candidates.
    """

    candidates = []

    row_count = len(df)

    for column in df.columns:

        series = df[column]
        clean_series = series.dropna()

        if clean_series.empty:
            continue

        name = str(column).lower().strip()

        unique_count = clean_series.nunique()
        unique_ratio = unique_count / len(clean_series)
        missing_ratio = series.isna().mean()

        reasons = []
        score = 0

        # ----------------------------------------------------
        # 1. Exclude unsuitable columns
        # ----------------------------------------------------

        if unique_count <= 1:
            continue

        if is_id_column(column, series):
            continue

        if is_date_column(series, column):
            continue

        if unique_ratio >= 0.98:
            continue

        # ----------------------------------------------------
        # 2. Check target-related column names
        # ----------------------------------------------------

        matching_keywords = [
            word for word in TARGET_KEYWORDS
            if re.search(
                r"(^|[_\s])" + re.escape(word) + r"($|[_\s])",
                name
            )
        ]

        if matching_keywords:
            score += 5

            reasons.append(
                "Target-related column name: "
                + ", ".join(matching_keywords)
            )

        # ----------------------------------------------------
        # 3. Identify task type
        # ----------------------------------------------------

        column_type = get_column_type(series)

        possible_task = get_task_type(
            clean_series,
            unique_count,
            row_count
        )

        if column_type == "numeric":

            if possible_task == "classification":
                score += 3

                reasons.append(
                    "Numeric column with a limited number "
                    "of distinct values"
                )

            else:
                score += 2

                reasons.append(
                    "Numeric column suitable for continuous prediction"
                )

        elif column_type == "categorical":

            score += 3

            reasons.append(
                "Categorical column suitable for classification"
            )

        # ----------------------------------------------------
        # 4. Evaluate unique-value ratio
        # ----------------------------------------------------

        if unique_ratio <= 0.05:
            score += 2

            reasons.append(
                "Low cardinality; potentially suitable as a target"
            )

        elif unique_ratio <= 0.30:
            score += 1

            reasons.append(
                "Moderate number of distinct values"
            )

        elif unique_ratio > 0.80:
            score -= 3

            reasons.append(
                "Very high cardinality; may not be a suitable target"
            )

        elif unique_ratio > 0.50:
            score -= 1

            reasons.append(
                "Relatively high cardinality"
            )

        # ----------------------------------------------------
        # 5. Evaluate missing values
        # ----------------------------------------------------

        if missing_ratio > 0.50:
            score -= 4

            reasons.append(
                "High percentage of missing values"
            )

        elif missing_ratio > 0.20:
            score -= 2

            reasons.append(
                "Some missing values"
            )

        elif missing_ratio > 0:
            score -= 1

            reasons.append(
                "Contains a small number of missing values"
            )

        else:
            score += 1

            reasons.append(
                "No missing values"
            )

        # ----------------------------------------------------
        # 6. Evaluate class distribution
        # ----------------------------------------------------

        class_distribution = {}

        if possible_task == "classification":

            class_distribution = calculate_class_distribution(
                clean_series
            )

            if len(class_distribution) == 2:

                values = list(class_distribution.values())

                if min(values) < 5:
                    score -= 2

                    reasons.append(
                        "Severely imbalanced binary classes"
                    )

                elif min(values) >= 20:
                    score += 1

                    reasons.append(
                        "Reasonably distributed binary classes"
                    )

            elif len(class_distribution) > 20:

                score -= 2

                reasons.append(
                    "Many classes; classification may be difficult"
                )

        # ----------------------------------------------------
        # 7. Evaluate dataset size
        # ----------------------------------------------------

        if row_count < MIN_ROWS_FOR_ML:
            score -= 3

            reasons.append(
                "Dataset has very few rows for reliable ML"
            )

        elif row_count >= 100:
            score += 1

            reasons.append(
                "Dataset has a reasonable number of rows"
            )

        # ----------------------------------------------------
        # Store candidate
        # ----------------------------------------------------

        candidates.append({
            "column": column,
            "column_type": column_type,
            "possible_task": possible_task,
            "unique_values": int(unique_count),
            "unique_ratio": round(unique_ratio, 3),
            "missing_percentage": round(
                missing_ratio * 100, 2
            ),
            "class_distribution": class_distribution,
            "score": score,
            "reasons": reasons
        })

    # Highest-scoring candidates first
    candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # Add ranking
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    return candidates


# ============================================================
# ALGORITHM RECOMMENDATIONS
# ============================================================

def recommend_regression_algorithms():

    return [
        {
            "algorithm": "Linear Regression",
            "reason": (
                "Provides a simple baseline for predicting "
                "continuous numeric values."
            )
        },
        {
            "algorithm": "Ridge Regression",
            "reason": (
                "Useful when input features are correlated "
                "and regularization is needed."
            )
        },
        {
            "algorithm": "Lasso Regression",
            "reason": (
                "Performs regularization and can reduce "
                "the influence of less useful features."
            )
        },
        {
            "algorithm": "Random Forest Regressor",
            "reason": (
                "Can capture nonlinear relationships "
                "and feature interactions."
            )
        },
        {
            "algorithm": "Gradient Boosting Regressor",
            "reason": (
                "Builds sequential trees to improve "
                "prediction performance."
            )
        }
    ]


def recommend_classification_algorithms():

    return [
        {
            "algorithm": "Logistic Regression",
            "reason": (
                "Provides a useful baseline for predicting "
                "categorical outcomes."
            )
        },
        {
            "algorithm": "Decision Tree Classifier",
            "reason": (
                "Learns decision rules and can model "
                "nonlinear relationships."
            )
        },
        {
            "algorithm": "Random Forest Classifier",
            "reason": (
                "Combines multiple decision trees to "
                "model complex patterns."
            )
        },
        {
            "algorithm": "K-Nearest Neighbors",
            "reason": (
                "Classifies observations using similarity "
                "between records."
            )
        },
        {
            "algorithm": "Naive Bayes",
            "reason": (
                "Provides a fast probabilistic "
                "classification baseline."
            )
        },
        {
            "algorithm": "Support Vector Machine",
            "reason": (
                "Can identify decision boundaries "
                "between different classes."
            )
        }
    ]


def recommend_clustering_algorithms():

    return [
        {
            "algorithm": "K-Means",
            "reason": (
                "Groups similar records into a chosen "
                "number of clusters."
            )
        },
        {
            "algorithm": "Hierarchical Clustering",
            "reason": (
                "Builds nested groups to explore "
                "similarities between observations."
            )
        },
        {
            "algorithm": "DBSCAN",
            "reason": (
                "Identifies dense groups and can "
                "detect isolated observations."
            )
        }
    ]


# ============================================================
# DATASET ANALYSIS AND ALGORITHM RECOMMENDATION
# ============================================================

def analyze_dataset(df, target_column=None, task_type=None):
    """
    Main algorithm recommendation engine.

    Automatically:
    - Inspects the dataset
    - Detects possible targets
    - Ranks target candidates
    - Selects the strongest candidate when confidence is sufficient
    - Determines the ML task
    - Recommends algorithms
    - Separates recommendations from actual model training

    target_column and task_type are optional.
    """

    # --------------------------------------------------------
    # 1. Validate dataset
    # --------------------------------------------------------

    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    if len(df.columns) < 2:
        raise ValueError(
            "Dataset needs at least two columns for this analysis."
        )

    if df.columns.duplicated().any():
        raise ValueError(
            "Dataset contains duplicate column names. "
            "Please make the column names unique."
        )

    if task_type is not None:

        task_type = task_type.lower().strip()

        if task_type not in SUPPORTED_TASKS:
            raise ValueError(
                "Task type must be regression, classification, "
                "or clustering."
            )

    if task_type == "clustering" and target_column is not None:
        raise ValueError(
            "Clustering is an unsupervised task and does not "
            "require a target column."
        )

    # --------------------------------------------------------
    # 2. Inspect columns
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    datetime_columns = [
        column for column in df.columns
        if is_date_column(df[column], column)
    ]

    # --------------------------------------------------------
    # 3. Detect possible targets
    # --------------------------------------------------------

    target_candidates = detect_target_candidates(df)

    selected_target = target_column
    selection_reason = None

    # --------------------------------------------------------
    # 4. Automatically select strongest candidate
    # --------------------------------------------------------

    if selected_target is None and task_type != "clustering":

        if target_candidates:

            best_candidate = target_candidates[0]

            # Require a meaningful score before selecting
            # a candidate automatically.
            if best_candidate["score"] >= 3:

                selected_target = best_candidate["column"]

                selection_reason = (
                    "Automatically selected the highest-ranked "
                    "target candidate."
                )

    # Validate explicitly selected target
    if selected_target is not None:

        if selected_target not in df.columns:
            raise ValueError(
                f"Target column '{selected_target}' does not exist."
            )

        if is_id_column(
            selected_target,
            df[selected_target]
        ):
            raise ValueError(
                "The selected target appears to be an ID column."
            )

        if is_date_column(
            df[selected_target],
            selected_target
        ):
            raise ValueError(
                "Date/time columns are not supported as prediction "
                "targets in this version."
            )

        if df[selected_target].nunique(dropna=True) <= 1:
            raise ValueError(
                "The selected target contains only one unique value."
            )

        if df[selected_target].isna().all():
            raise ValueError(
                "The selected target contains only missing values."
            )

    # --------------------------------------------------------
    # 5. Determine ML task
    # --------------------------------------------------------

    detected_task = task_type

    if detected_task is None and selected_target is not None:

        target_series = df[selected_target].dropna()

        detected_task = get_task_type(
            target_series,
            target_series.nunique(),
            len(df)
        )

    # --------------------------------------------------------
    # 6. Generate recommendations
    # --------------------------------------------------------

    recommendations = []

    if detected_task == "regression":

        if selected_target is None:
            raise ValueError(
                "A target column is required for regression."
            )

        if not pd.api.types.is_numeric_dtype(
            df[selected_target]
        ):
            raise ValueError(
                "Regression requires a numeric target column."
            )

        recommendations = recommend_regression_algorithms()

    elif detected_task == "classification":

        if selected_target is None:
            raise ValueError(
                "A target column is required for classification."
            )

        recommendations = recommend_classification_algorithms()

    elif detected_task == "clustering":

        # Clustering needs usable input features.
        usable_numeric = [
            column for column in numeric_columns
            if column != selected_target
            and df[column].nunique(dropna=True) > 1
        ]

        if len(usable_numeric) < 2:
            raise ValueError(
                "At least two non-constant numeric columns "
                "are required for clustering in this version."
            )

        recommendations = recommend_clustering_algorithms()

        selection_reason = (
            "Clustering was selected because no supervised "
            "target was selected or the user requested clustering."
        )

    else:

        # No reliable target was identified.
        # Do not invent a supervised prediction task.
        detected_task = "Needs user selection"

        selection_reason = (
            "No sufficiently strong target candidate was found. "
            "Choose a target column or select clustering."
        )

        if len(numeric_columns) >= 2:

            recommendations = recommend_clustering_algorithms()

            selection_reason += (
                " Clustering algorithms are suggested as an "
                "unsupervised alternative."
            )

    # --------------------------------------------------------
    # 7. Add recommendation status
    # --------------------------------------------------------

    for item in recommendations:

        item["status"] = "Recommended - Not Trained"

    # --------------------------------------------------------
    # 8. Calculate dataset information
    # --------------------------------------------------------

    dataset_information = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_column_count": len(numeric_columns),
        "categorical_column_count": len(categorical_columns),
        "datetime_column_count": len(datetime_columns),
        "total_missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum())
    }

    # --------------------------------------------------------
    # 9. Return complete analysis
    # --------------------------------------------------------

    return {

        "dataset_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        },

        "dataset_information": dataset_information,

        "numeric_columns": numeric_columns,

        "categorical_columns": categorical_columns,

        "datetime_columns": datetime_columns,

        "target_candidates": target_candidates,

        "selected_target": selected_target,

        "target_selection_reason": selection_reason,

        "detected_task": detected_task,

        "recommendations": recommendations,

        "training_performed": False,

        "model_results": None,

        "message": (
            "Algorithms are recommended based on dataset "
            "characteristics. No machine learning model "
            "has been trained or evaluated."
        )
    }
