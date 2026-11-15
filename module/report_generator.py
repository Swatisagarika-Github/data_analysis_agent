
import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)


def create_data_quality(df):
    """Calculate dataset quality information."""

    total_missing = int(df.isnull().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    total_cells = df.shape[0] * df.shape[1]

    complete_rows = int(
        df.dropna().shape[0]
    )

    quality_data = [
        ["Metric", "Value"],
        ["Total Rows", len(df)],
        ["Total Columns", len(df.columns)],
        ["Missing Values", total_missing],
        ["Duplicate Rows", duplicate_rows],
        ["Complete Rows", complete_rows],
        [
            "Missing Value Percentage",
            round((total_missing / total_cells) * 100, 2)
            if total_cells else 0
        ]
    ]

    return quality_data


def create_metric_table(df):
    """Generate numerical summary statistics."""

    numeric_df = df.select_dtypes(include="number")

    metric_data = [
        ["Column", "Mean", "Median", "Minimum", "Maximum"]
    ]

    for column in numeric_df.columns:

        metric_data.append([
            str(column),
            round(numeric_df[column].mean(), 2),
            round(numeric_df[column].median(), 2),
            round(numeric_df[column].min(), 2),
            round(numeric_df[column].max(), 2)
        ])

    return metric_data


def create_charts(df, chart_folder):
    """Generate charts for the PDF report."""

    os.makedirs(chart_folder, exist_ok=True)

    chart_paths = []

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numerical distribution chart
    if numeric_columns:

        column = numeric_columns[0]

        plt.figure(figsize=(7, 4))
        df[column].dropna().hist(bins=20)

        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.tight_layout()

        path = os.path.join(
            chart_folder,
            "numerical_distribution.png"
        )

        plt.savefig(path)
        plt.close()

        chart_paths.append(path)

    # Correlation heatmap
    if len(numeric_columns) >= 2:

        correlation = df[numeric_columns].corr()

        plt.figure(figsize=(7, 5))

        plt.imshow(
            correlation,
            cmap="coolwarm",
            interpolation="nearest"
        )

        plt.colorbar()
        plt.xticks(
            range(len(correlation.columns)),
            correlation.columns,
            rotation=45,
            ha="right"
        )

        plt.yticks(
            range(len(correlation.columns)),
            correlation.columns
        )

        plt.title("Correlation Heatmap")
        plt.tight_layout()

        path = os.path.join(
            chart_folder,
            "correlation_heatmap.png"
        )

        plt.savefig(path)
        plt.close()

        chart_paths.append(path)

    # Categorical analysis chart
    if categorical_columns:

        column = categorical_columns[0]

        top_categories = (
            df[column]
            .value_counts()
            .head(10)
        )

        plt.figure(figsize=(7, 4))

        top_categories.plot(kind="bar")

        plt.title(f"Top Categories in {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        path = os.path.join(
            chart_folder,
            "category_analysis.png"
        )

        plt.savefig(path)
        plt.close()

        chart_paths.append(path)

    return chart_paths


def get_implemented_techniques(df, cleaning_report=None):
    """Identify techniques actually used by the current pipeline."""

    techniques = []

    cleaning_report = cleaning_report or {}
    actions = cleaning_report.get("cleaning_actions", [])

    # Cleaning techniques: identify them from recorded actions.
    for action in actions:
        action_lower = action.lower()

        if "using mean" in action_lower:
            techniques.append([
                "Mean Imputation",
                "Fill missing numerical values",
                "Executed"
            ])

        elif "using median" in action_lower:
            techniques.append([
                "Median Imputation",
                "Fill missing numerical values",
                "Executed"
            ])

        elif "using mode" in action_lower:
            techniques.append([
                "Mode Imputation",
                "Fill missing categorical values",
                "Executed"
            ])

        elif "missing values" in action_lower and (
            "dropped" in action_lower or "drop" in action_lower
        ):
            techniques.append([
                "Missing-Value Row Removal",
                "Remove rows containing missing values",
                "Executed"
            ])

        if "duplicate rows" in action_lower:
            techniques.append([
                "Duplicate Removal",
                "Remove repeated records",
                "Executed"
            ])

        if "standardized category" in action_lower:
            techniques.append([
                "Category Standardization",
                "Normalize category text",
                "Executed"
            ])

        if "converted" in action_lower and "datetime" in action_lower:
            techniques.append([
                "Datetime Conversion",
                "Convert selected columns to dates",
                "Executed"
            ])

        if "outlier" in action_lower and "removed" in action_lower:
            techniques.append([
                "IQR Outlier Removal",
                "Remove rows containing selected-column outliers",
                "Executed"
            ])

    # IQR detection is recorded for each selected numeric column.
    if cleaning_report.get("outliers"):
        techniques.append([
            "IQR Outlier Detection",
            "Detect outliers using the 1.5 × IQR rule",
            "Executed"
        ])

    # EDA techniques supported by the current report generator.
    numeric_columns = df.select_dtypes(include="number").columns
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns

    if len(numeric_columns) >= 1:
        techniques.append([
            "Histogram",
            "Visualize numerical data distribution",
            "Executed in PDF"
        ])

    if len(numeric_columns) >= 2:
        techniques.append([
            "Pearson Correlation",
            "Measure linear relationships between numerical columns",
            "Executed in PDF"
        ])

        techniques.append([
            "Correlation Heatmap",
            "Visualize the correlation matrix",
            "Executed in PDF"
        ])

    if len(categorical_columns) >= 1:
        techniques.append([
            "Category Frequency Analysis",
            "Count the frequency of category values",
            "Executed in PDF"
        ])

        techniques.append([
            "Bar Chart",
            "Visualize the most frequent categories",
            "Executed in PDF"
        ])

    return techniques


def recommend_ml_algorithms(df):
    """Suggest possible ML algorithms from dataset characteristics."""

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    recommendations = []

    if len(numeric_columns) >= 2:
        recommendations.extend([
            [
                "Linear Regression",
                "Possible when predicting a numerical target",
                "Not executed"
            ],
            [
                "Random Forest Regressor",
                "Can model nonlinear numerical relationships",
                "Not executed"
            ],
            [
                "K-Means Clustering",
                "Can group records using numerical features",
                "Not executed"
            ],
            [
                "Isolation Forest",
                "Can identify unusual records",
                "Not executed"
            ]
        ])

    if categorical_columns:
        recommendations.extend([
            [
                "Logistic Regression",
                "Possible for a binary or suitable categorical target",
                "Not executed"
            ],
            [
                "Decision Tree Classifier",
                "Can classify records using a labelled target",
                "Not executed"
            ],
            [
                "Random Forest Classifier",
                "Can classify records using a labelled target",
                "Not executed"
            ]
        ])

    if len(numeric_columns) >= 3:
        recommendations.append([
            "PCA",
            "Can reduce the number of numerical features",
            "Not executed"
        ])

    if not recommendations:
        recommendations.append([
            "No suitable recommendation yet",
            "More usable numerical or categorical features are needed",
            "Not applicable"
        ])

    return recommendations

def generate_pdf_report(
            df,
            eda_report,
            ai_insight=None,
            cleaning_report=None,
            output_path="reports/analysis_report.pdf"
    ):


    """Generate a complete PDF analysis report."""

    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    output_folder = os.path.dirname(output_path)

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    chart_folder = os.path.join(
        output_folder if output_folder else ".",
        "report_charts"
    )

    styles = getSampleStyleSheet()

    story = []

    # PDF document
    document = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    # Title
    story.append(
        Paragraph(
            "AI-Powered Data Analytics Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Automated Dataset Analysis and Business Insights",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 20))

    # Dataset overview
    story.append(
        Paragraph(
            "1. Dataset Overview",
            styles["Heading1"]
        )
    )

    overview_data = [
        ["Description", "Value"],
        ["Total Records", len(df)],
        ["Total Columns", len(df.columns)],
        ["Numerical Columns", len(df.select_dtypes(include="number").columns)],
        [
            "Categorical Columns",
            len(df.select_dtypes(include=["object", "category", "bool"]).columns)
        ]
    ]

    overview_table = Table(
        overview_data,
        repeatRows=1
    )

    overview_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7)
        ])
    )

    story.append(overview_table)
    story.append(Spacer(1, 20))

    # Data quality
    story.append(
        Paragraph(
            "2. Data Quality Findings",
            styles["Heading1"]
        )
    )

    quality_table = Table(
        create_data_quality(df),
        repeatRows=1
    )

    quality_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING", (0, 0), (-1, -1), 7)
        ])
    )

    story.append(quality_table)
    story.append(Spacer(1, 20))

    # Numerical metrics
    story.append(
        Paragraph(
            "3. Important Statistical Metrics",
            styles["Heading1"]
        )
    )

    metric_data = create_metric_table(df)

    if len(metric_data) > 1:

        metric_table = Table(
            metric_data,
            repeatRows=1
        )

        metric_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("PADDING", (0, 0), (-1, -1), 6)
            ])
        )

        story.append(metric_table)

    else:
        story.append(
            Paragraph(
                "No numerical columns available.",
                styles["BodyText"]
            )
        )

    story.append(Spacer(1, 20))

    # Algorithm and technique information
    story.append(
        Paragraph(
            "4. Algorithms and Techniques Implemented",
            styles["Heading1"]
        )
    )

    implemented = get_implemented_techniques(
        df,
        cleaning_report
    )

    implemented_data = [
        ["Technique", "Purpose", "Execution Status"]
    ]

    implemented_data.extend(implemented)

    if len(implemented_data) > 1:
        implemented_table = Table(
            implemented_data,
            colWidths=[1.65 * inch, 2.8 * inch, 1.1 * inch],
            repeatRows=1
        )

        implemented_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("PADDING", (0, 0), (-1, -1), 6)
            ])
        )

        story.append(implemented_table)

    else:
        story.append(
            Paragraph(
                "No recorded cleaning techniques were available. "
                "Only the analysis techniques executed in this report "
                "are listed when applicable.",
                styles["BodyText"]
            )
        )

    story.append(Spacer(1, 15))

    # Recommended machine learning algorithms
    story.append(
        Paragraph(
            "5. Recommended Machine Learning Algorithms",
            styles["Heading1"]
        )
    )

    recommendations = recommend_ml_algorithms(df)

    recommendation_data = [
        ["Algorithm", "Reason for Recommendation", "Status"]
    ]

    recommendation_data.extend(recommendations)

    recommendation_table = Table(
        recommendation_data,
        colWidths=[1.65 * inch, 2.8 * inch, 1.1 * inch],
        repeatRows=1
    )

    recommendation_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(recommendation_table)
    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Note: Recommended algorithms are suggestions based on "
            "the available column types. They have not been trained "
            "or evaluated. Select a target column and run model "
            "training before reporting model performance.",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 20))

    # AI analysis summary
    story.append(
        Paragraph(
            "6. AI-Generated Analysis and Insights",
            styles["Heading1"]
        )
    )

    if ai_insight:

        for line in str(ai_insight).splitlines():

            line = line.strip()

            if line:

                story.append(
                    Paragraph(
                        escape(line),
                        styles["BodyText"]
                    )
                )

                story.append(Spacer(1, 5))

    else:

        story.append(
            Paragraph(
                "AI insights were not provided.",
                styles["BodyText"]
            )
        )

    story.append(Spacer(1, 20))

    # Charts
    story.append(
        Paragraph(
            "7. Data Visualizations",
            styles["Heading1"]
        )
    )

    chart_paths = create_charts(
        df,
        chart_folder
    )

    if chart_paths:

        for chart_path in chart_paths:

            story.append(
                Image(
                    chart_path,
                    width=6 * inch,
                    height=3.5 * inch
                )
            )

            story.append(Spacer(1, 15))

    else:

        story.append(
            Paragraph(
                "No suitable charts could be generated.",
                styles["BodyText"]
            )
        )

    # Conclusion
    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "8. Conclusion",
            styles["Heading1"]
        )
    )

    story.append(
        Paragraph(
            "This report presents an automated analysis of the uploaded "
            "dataset, including data quality findings, statistical metrics, "
            "visualizations and AI-generated insights.",
            styles["BodyText"]
        )
    )

    # Build PDF
    document.build(story)

    return output_path
