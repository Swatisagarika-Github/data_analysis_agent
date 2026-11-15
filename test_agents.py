import pandas as pd

from module.data_understanding import understand_dataset
from module.data_cleaning import clean_dataset
from agents.visualization_agent import run_visualization_agent


df = pd.DataFrame({
    "Product": ["Laptop", "Mobile", "Tablet", "Laptop", "Mobile"],
    "Category": ["Electronics", "Electronics", "Electronics",
                 "Electronics", "Electronics"],
    "Sales": [50000, 20000, 15000, 55000, 25000],
    "Quantity": [2, 3, 1, 2, 4]
})


print("\n1. DATA UNDERSTANDING")
understanding_result = understand_dataset(df)
print("Type:", type(understanding_result))
print("Result:", understanding_result)


print("\n2. DATA CLEANING")
cleaned_df, cleaning_report = clean_dataset(df)
print("Cleaned DataFrame:")
print(cleaned_df)
print("Cleaning Report:")
print(cleaning_report)


print("\n3. VISUALIZATION")
visualization_result = run_visualization_agent(df)
print("Type:", type(visualization_result))
print("Result:", visualization_result)