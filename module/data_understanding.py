
import pandas as pd


def understand_dataset(df):

    # Dataset size
    total_rows = int(df.shape[0])
    total_columns = int(df.shape[1])

    # Data types as ordinary Python strings
    data_types = {
        str(column): str(dtype)
        for column, dtype in df.dtypes.items()
    }

    # Missing values
    missing_values = {
        str(column): int(count)
        for column, count in df.isnull().sum().items()
    }

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Unique values
    unique_values = {
        str(column): int(count)
        for column, count in df.nunique().items()
    }

    # Numerical columns
    numerical_columns = (
        df.select_dtypes(include="number").columns.tolist()
    )

    # Categorical columns
    categorical_columns = (
        df.select_dtypes(
            include=["object", "category", "string", "bool"]
        ).columns.tolist()
    )

    # Convert column names to strings
    numerical_columns = [
        str(column) for column in numerical_columns
    ]

    categorical_columns = [
        str(column) for column in categorical_columns
    ]

    profile = {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "data_types": data_types,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "unique_values": unique_values,
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns
    }

    return profile
