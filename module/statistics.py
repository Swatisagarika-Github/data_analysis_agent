
import pandas as pd


def numerical_analysis(df):
    numerical_df = df.select_dtypes(include="number")

    results = {}

    for column in numerical_df.columns:

        values = numerical_df[column].dropna()

        if values.empty:
            continue

        mode_values = values.mode().tolist()

        results[column] = {
            "count": int(values.count()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "mode": mode_values,
            "standard_deviation": float(values.std()),
            "variance": float(values.var()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "Q1": float(values.quantile(0.25)),
            "Q2": float(values.quantile(0.50)),
            "Q3": float(values.quantile(0.75))
        }

    return results
