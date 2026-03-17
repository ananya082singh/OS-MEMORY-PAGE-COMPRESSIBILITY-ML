import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("phase2_stage1_v2_results.csv")

features = [
"feat_distinct",
"feat_max_freq_ratio",
"feat_zero_ratio",
"feat_run_count",
"feat_entropy"
]

X = df[features]
y = df["label"].map({"compressible":1,"incompressible":0})

model = RandomForestClassifier()
model.fit(X,y)

importances = model.feature_importances_

plt.bar(features, importances)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.savefig("feature_importance.png")
print("Feature importance plot saved as feature_importance.png")