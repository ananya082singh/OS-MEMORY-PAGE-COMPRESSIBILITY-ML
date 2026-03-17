import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("phase2_stage1_v2_results.csv")

features = [
"feat_distinct",
"feat_max_freq_ratio",
"feat_zero_ratio",
"feat_run_count"
]

X = df[features]
y = df["label"].map({"compressible":1,"incompressible":0})

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = XGBClassifier(eval_metric="logloss")

param_grid = {
    "n_estimators":[100,200,300],
    "max_depth":[3,5,7],
    "learning_rate":[0.01,0.05,0.1],
    "subsample":[0.8,1]
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train,y_train)

best_model = grid.best_estimator_

preds = best_model.predict(X_test)

print("Best parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test,preds))