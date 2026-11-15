
import json

from ai.llm_service import generate_ai_response


def make_json_safe(data):
    """
    Convert Python objects into JSON-compatible values.
    Handles dictionaries, lists, DataFrames and NumPy values.
    """
    if isinstance(data, dict):
        return {
            str(key): make_json_safe(value)
            for key, value in data.items()
        }

    if isinstance(data, list):
        return [make_json_safe(item) for item in data]

    if isinstance(data, tuple):
        return [make_json_safe(item) for item in data]

    if hasattr(data, "to_dict"):
        return make_json_safe(data.to_dict())

    if hasattr(data, "item"):
        try:
            return data.item()
        except (ValueError, TypeError):
            pass

    if data is None or isinstance(data, (str, int, float, bool)):
        return data

    return str(data)


def run_insight_agent(eda_report):
    """
    Generate AI-powered insights from the EDA report.
    If Gemini is unavailable, use a Python-based fallback.
    """

    if not eda_report:
        return "No EDA report available to generate insights."

    safe_eda_report = make_json_safe(eda_report)

    prompt = f"""
You are an expert Data Analyst and AI assistant.

Analyze the following Exploratory Data Analysis (EDA) report.

EDA REPORT:
{json.dumps(safe_eda_report, indent=2, ensure_ascii=False)}

Generate a clear and professional analysis containing:

1. Dataset Overview
2. Important Findings
3. Numerical Analysis
4. Categorical Analysis
5. Correlation Analysis
6. Business Insights
7. Anomalies or Potential Issues
8. Actionable Recommendations

Instructions:
- Explain findings in simple language.
- Use actual values from the EDA report wherever available.
- Do not invent statistics or results.
- If information is unavailable, mention that clearly.
- Format the response with clear headings and bullet points.
"""

    try:
        print("Generating AI insights using Gemini...")

        insight = generate_ai_response(prompt)

        if not insight:
            raise ValueError("Gemini returned an empty response.")

        print("AI insights generated successfully.")

        return insight

    except Exception as error:
        print(f"Gemini API unavailable: {error}")
        print("Using Python-based fallback insights...")

        fallback_insight = (
            "AI INSIGHT REPORT - FALLBACK MODE\n\n"
            "Gemini is temporarily unavailable. "
            "The following information is the EDA report "
            "generated through Python analysis. "
            "It is not AI-generated recommendations.\n\n"
            "EDA ANALYSIS RESULTS:\n\n"
            + json.dumps(
                safe_eda_report,
                indent=2,
                ensure_ascii=False
            )[:15000]
        )

        return fallback_insight
