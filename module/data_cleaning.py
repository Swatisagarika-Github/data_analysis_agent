
import pandas as pd


def clean_dataset(
    df,
    numeric_strategy="Median",
    categorical_strategy="Mode",
    remove_duplicates=True,
    outlier_action="Keep",
    outlier_columns=None,
    standardize_categories=False,
    category_columns=None,
    category_format="Trim spaces only",
    date_columns=None
):

    # Keep original dataset unchanged
    cleaned_df = df.copy()

    # Before cleaning information
    report = {
        "rows_before": int(len(cleaned_df)),
        "columns_before": int(len(cleaned_df.columns)),
        "missing_before": int(cleaned_df.isnull().sum().sum()),
        "duplicates_before": int(cleaned_df.duplicated().sum()),
        "outliers": {},
        "converted_dates": {},
        "cleaning_actions": []
    }

    # --------------------------------
    # 1. Handle Missing Values
    # --------------------------------

    if (
        numeric_strategy == "Drop rows with missing values"
        or categorical_strategy == "Drop rows with missing values"
    ):

        before = len(cleaned_df)

        cleaned_df = cleaned_df.dropna()

        report["cleaning_actions"].append(
            f"Dropped {before - len(cleaned_df)} rows containing missing values."
        )

    else:

        for column in cleaned_df.columns:

            if cleaned_df[column].isnull().sum() == 0:
                continue

            if pd.api.types.is_numeric_dtype(cleaned_df[column]):

                if numeric_strategy == "Mean":
                    value = cleaned_df[column].mean()

                elif numeric_strategy == "Median":
                    value = cleaned_df[column].median()

                else:
                    continue

                if pd.notna(value):
                    cleaned_df[column] = cleaned_df[column].fillna(value)

                    report["cleaning_actions"].append(
                        f"Filled missing values in {column} using {numeric_strategy}."
                    )

            else:

                if categorical_strategy == "Mode":

                    mode_values = cleaned_df[column].mode()

                    if not mode_values.empty:
                        fill_value = mode_values.iloc[0]
                    else:
                        fill_value = "Unknown"

                elif categorical_strategy == "Fill with Unknown":
                    fill_value = "Unknown"

                else:
                    continue

                cleaned_df[column] = cleaned_df[column].fillna(fill_value)

                report["cleaning_actions"].append(
                    f"Filled missing values in {column} using {categorical_strategy}."
                )

    # --------------------------------
    # 2. Remove Duplicate Records
    # --------------------------------

    if remove_duplicates:

        duplicate_count = int(cleaned_df.duplicated().sum())

        cleaned_df = cleaned_df.drop_duplicates()

        report["cleaning_actions"].append(
            f"Removed {duplicate_count} duplicate rows."
        )

    # --------------------------------
    # 3. Detect Outliers using IQR
    # --------------------------------

    if outlier_columns is None:
        outlier_columns = []

    outlier_mask = pd.Series(
        False,
        index=cleaned_df.index
    )

    for column in outlier_columns:

        if column not in cleaned_df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(cleaned_df[column]):
            continue

        values = cleaned_df[column].dropna()

        if values.empty:
            continue

        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)

        IQR = Q3 - Q1

        lower_limit = Q1 - (1.5 * IQR)
        upper_limit = Q3 + (1.5 * IQR)

        column_outliers = (
            (cleaned_df[column] < lower_limit)
            | (cleaned_df[column] > upper_limit)
        )

        outlier_count = int(column_outliers.sum())

        report["outliers"][column] = {
            "count": outlier_count,
            "lower_limit": float(lower_limit),
            "upper_limit": float(upper_limit)
        }

        if outlier_action == "Remove rows":
            outlier_mask = outlier_mask | column_outliers.fillna(False)

    if outlier_action == "Remove rows":

        outlier_count = int(outlier_mask.sum())

        cleaned_df = cleaned_df.loc[~outlier_mask].copy()

        report["cleaning_actions"].append(
            f"Removed {outlier_count} rows containing selected-column outliers."
        )

    # --------------------------------
    # 4. Standardize Categories
    # --------------------------------

    if standardize_categories and category_columns:

        for column in category_columns:

            if column not in cleaned_df.columns:
                continue

            cleaned_df[column] = (
                cleaned_df[column]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )

            if category_format == "Title Case":
                cleaned_df[column] = cleaned_df[column].str.title()

            elif category_format == "Lowercase":
                cleaned_df[column] = cleaned_df[column].str.lower()

            report["cleaning_actions"].append(
                f"Standardized category values in {column}."
            )

    # --------------------------------
    # 5. Convert Selected Columns to Dates
    # --------------------------------

    if date_columns:

        for column in date_columns:

            if column not in cleaned_df.columns:
                continue

            before_invalid = cleaned_df[column].isna().sum()

            converted = pd.to_datetime(
                cleaned_df[column],
                errors="coerce",
                format="mixed",
                dayfirst=True
            )

            new_invalid = int(converted.isna().sum() - before_invalid)

            cleaned_df[column] = converted

            report["converted_dates"][column] = new_invalid

            report["cleaning_actions"].append(
                f"Converted {column} to datetime. "
                f"New invalid dates: {new_invalid}."
            )

    # --------------------------------
    # 6. After Cleaning Information
    # --------------------------------

    report["rows_after"] = int(len(cleaned_df))
    report["columns_after"] = int(len(cleaned_df.columns))
    report["missing_after"] = int(cleaned_df.isnull().sum().sum())
    report["duplicates_after"] = int(cleaned_df.duplicated().sum())

    return cleaned_df, report
