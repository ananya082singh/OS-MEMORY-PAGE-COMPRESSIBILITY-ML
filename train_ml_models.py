import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

print("="*60)
print("TRAINING MACHINE LEARNING MODELS")
print("="*60)

# Load dataset
df = pd.read_csv("phase2_stage1_v2_results.csv")
print("\nMissing values per column:")
print(df.isna().sum())

df = df.fillna(0)

print("\nDataset Loaded")
print("Total samples:", len(df))

# Select features
features = [
    "feat_distinct",
    "feat_max_freq_ratio",
    "feat_zero_ratio",
    "feat_run_count",
    "feat_entropy"
]

X = df[features]

# Convert label to numeric
y = df["label"].map({
    "compressible": 1,
    "incompressible": 0
})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200),
    "SVM": SVC(),
    "XGBoost": XGBClassifier()
}

# Train and evaluate
for name, model in models.items():
    
    print("\n" + "="*50)
    print("MODEL:", name)
    print("="*50)
    
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    
    print("\nAccuracy:", round(acc,4))
    
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

print("\n✅ ML training complete!")