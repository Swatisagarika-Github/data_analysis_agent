
import pandas as pd
from module.algorithm_recommender import analyze_dataset

# Sample dataset for testing
data = {
    "Age": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
    "StudyHours": [2, 3, 4, 5, 6, 7, 8, 5, 6, 9],
    "Attendance": [60, 65, 70, 75, 80, 85, 90, 78, 82, 95],
    "PlacementStatus": [
        "Not Placed", "Not Placed", "Not Placed",
        "Placed", "Placed", "Placed", "Placed",
        "Placed", "Placed", "Placed"
    ]
}

df = pd.DataFrame(data)

# Analyze dataset
result = analyze_dataset(df)

print("\n========== DATASET INFORMATION ==========")
print(result["dataset_information"])

print("\n========== TARGET CANDIDATES ==========")
for candidate in result["target_candidates"]:
    print(candidate)

print("\n========== SELECTED TARGET ==========")
print(result["selected_target"])

print("\n========== DETECTED TASK ==========")
print(result["detected_task"])

print("\n========== ALGORITHM RECOMMENDATIONS ==========")
for algorithm in result["recommendations"]:
    print(algorithm)

print("\nTraining performed:", result["training_performed"])
