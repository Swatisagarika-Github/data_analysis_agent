
import pandas as pd


def categorical_analysis(df):
    categorical_df = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    )

    results = {}

    for column in categorical_df.columns:

        counts = categorical_df[column].value_counts(
            dropna=False
        )

        results[column] = {
            "unique_categories": int(df[column].nunique()),
            "frequency": {
                str(key): int(value)
                for key, value in counts.items()
            },
            "most_common": (
                str(counts.index[0]) if not counts.empty else None
            )
        }

    return results


def correlation_analysis(df):
    numerical_df = df.select_dtypes(include="number")

    if numerical_df.shape[1] < 2:
        return pd.DataFrame()

    return numerical_df.corr()


def category_comparison(df):
    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    results = {}

    for category in categorical_columns:
        for numerical in numerical_columns:

            comparison = df.groupby(
                category,
                dropna=False
            )[numerical].mean()

            results[f"{numerical} by {category}"] = comparison

    return results


def time_trend_analysis(df, date_columns=None):
    results = {}

    if date_columns is None:
        date_columns = df.select_dtypes(
            include=["datetime"]
        ).columns.tolist()

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    for date_column in date_columns:

        if date_column not in df.columns:
            continue

        temp_df = df[[date_column] + numerical_columns].copy()

        temp_df[date_column] = pd.to_datetime(
            temp_df[date_column],
            errors="coerce"
        )

        temp_df = temp_df.dropna(subset=[date_column])

        if temp_df.empty or not numerical_columns:
            continue

        temp_df = temp_df.set_index(date_column)

        results[date_column] = (
            temp_df[numerical_columns]
            .resample("MS")
            .mean()
        )

    return results
