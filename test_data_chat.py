
import pandas as pd

from agents.data_chat_agent import answer_data_question


def load_test_dataset():

    print("\n========== AI DATA CHAT ==========")

    file_path = input(
        "Enter your dataset path: "
    ).strip().strip('"')

    try:

        if file_path.lower().endswith(".csv"):
            df = pd.read_csv(file_path)

        elif file_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)

        else:
            print("Unsupported format. Use CSV or Excel.")
            return None

        if df.empty:
            print("Dataset is empty.")
            return None

        print("\nDataset loaded successfully!")

        print("\nDataset Shape:", df.shape)

        print("\nAvailable Columns:")
        print(df.columns.tolist())

        print("\nDataset Preview:")
        print(df.head())

        return df

    except Exception as error:

        print("Dataset loading failed:", error)
        return None


def start_data_chat():

    df = load_test_dataset()

    if df is None:
        return

    print("\n========== DATA CHAT STARTED ==========")
    print("Ask questions about your dataset.")
    print("Type 'exit' to stop the chat.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in ["exit", "quit", "stop"]:
            print("\nAI Data Chat stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:

            print("\nAI is analyzing your question...")

            answer = answer_data_question(
                df=df,
                question=question
            )

            print("\nAI Data Analyst:")
            print(answer)

        except Exception as error:

            print("\nError:", error)


if __name__ == "__main__":
    start_data_chat()
