
import pandas as pd

from module.model_evaluator import evaluate_models

# Sample classification dataset
df = pd.DataFrame({
    "StudyHours": [2, 3, 4, 5, 6, 7, 8, 9, 3, 6,
                   4, 8, 9, 2, 7, 5, 6, 8, 3, 9],
    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 55, 72,
                   62, 85, 95, 45, 82, 68, 74, 88, 52, 98],
    "PreviousScore": [40, 45, 50, 55, 60, 65, 70, 85, 42, 62,
                      51, 78, 90, 35, 75, 58, 67, 82, 44, 95],
    "PlacementStatus": [
        0, 0, 0, 0, 1, 1, 1, 1, 0, 1,
        0, 1, 1, 0, 1, 0, 1, 1, 0, 1
    ]
})

# Run model evaluation
result = evaluate_models(
    df=df,
    target_column="PlacementStatus"
)

print("\nMODEL PERFORMANCE COMPARISON")
print("--------------------------------")

comparison = pd.DataFrame(result["results"])

print(comparison.to_string(index=False))

print("\nBEST PERFORMING ALGORITHM")
print("--------------------------------")

print("Algorithm:", result["best_algorithm"])
print(
    "Accuracy:",
    result["best_performance"]["accuracy"],
    "%"
)
