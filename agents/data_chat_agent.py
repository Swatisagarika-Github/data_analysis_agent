
import pandas as pd

from ai.llm_service import generate_ai_response


def build_dataset_context(df):
    """Build a compact summary of the uploaded dataset."""

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    context = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": {},
        "categorical_summary": {}
    }

    # Numeric statistics
    if numeric_columns:
        context["numeric_summary"] = (
            df[numeric_columns].describe().round(2).to_dict()
        )

    # Top values in categorical columns
    for column in categorical_columns[:20]:
        context["categorical_summary"][column] = (
            df[column]
            .value_counts(dropna=False)
            .head(10)
            .to_dict()
        )

    return context


def answer_data_question(df, question):
    """Answer a user's question using dataset information."""

    if df is None or df.empty:
        raise ValueError("No dataset is available for analysis.")

    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    dataset_context = build_dataset_context(df)

    prompt = f"""
You are an intelligent AI Data Analyst.

Answer the user's question using only the dataset information
provided below.

DATASET INFORMATION:
{dataset_context}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Give a clear and beginner-friendly answer.
2. Use only the supplied dataset information.
3. Do not invent numbers, statistics, or column names.
4. If the required information is unavailable, explain that.
5. If the question requires a calculation that cannot be
   performed from the supplied summary, say that exact
   calculation is not yet supported.
6. Keep the answer concise and structured.
"""

    return generate_ai_response(prompt)
