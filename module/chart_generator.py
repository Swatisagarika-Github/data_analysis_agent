
import pandas as pd
import plotly.express as px
import streamlit as st


def generate_chart(df, chart_type, x=None, y=None):

    if chart_type == "Bar Chart":
        chart_data = df.groupby(
            x, dropna=False
        )[y].sum().reset_index()

        fig = px.bar(
            chart_data,
            x=x,
            y=y,
            title=f"{y} by {x}",
            text_auto=True
        )

    elif chart_type == "Line Chart":
        chart_data = df[[x, y]].copy()
        chart_data[x] = pd.to_datetime(
            chart_data[x], errors="coerce"
        )
        chart_data = chart_data.dropna(subset=[x])
        chart_data = chart_data.groupby(x, as_index=False)[y].sum()
        chart_data = chart_data.sort_values(x)

        fig = px.line(
            chart_data,
            x=x,
            y=y,
            markers=True,
            title=f"{y} Over Time"
        )

    elif chart_type == "Histogram":
        fig = px.histogram(
            df,
            x=x,
            title=f"Distribution of {x}",
            nbins=20
        )

    elif chart_type == "Scatter Plot":
        fig = px.scatter(
            df,
            x=x,
            y=y,
            title=f"{y} vs {x}"
        )

    elif chart_type == "Pie Chart":
        chart_data = (
            df[x].value_counts(dropna=False)
            .rename_axis(x)
            .reset_index(name="Count")
        )

        fig = px.pie(
            chart_data,
            names=x,
            values="Count",
            title=f"Distribution of {x}"
        )

    elif chart_type == "Correlation Heatmap":
        numerical_df = df.select_dtypes(include="number")

        if numerical_df.shape[1] < 2:
            st.warning("At least two numerical columns are required.")
            return

        correlation = numerical_df.corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap"
        )

    else:
        st.warning("Unsupported chart type.")
        return

    fig.update_layout(
        height=450,
        template="plotly_white"
    )

    st.plotly_chart(fig, width="stretch")
