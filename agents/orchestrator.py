
from agents.graph import analytics_graph


def run_analytics_workflow(df, cleaning_approved=False):

    if df is None or df.empty:
        raise ValueError("Uploaded dataset is empty.")

    initial_state = {
        "raw_df": df,
        "cleaning_approved": cleaning_approved,
        "current_step": "workflow_started"
    }

    result = analytics_graph.invoke(initial_state)

    return result
