import matplotlib.pyplot as plt

models = [
    "Logistic Regression",
    "Random Forest",
    "SVM",
    "XGBoost"
]

accuracies = [
    0.977,
    0.979,
    0.976,
    0.978
]

plt.figure(figsize=(8,5))
plt.bar(models, accuracies)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Model")

plt.ylim(0.95,1.0)

for i,v in enumerate(accuracies):
    plt.text(i,v+0.0005,str(round(v,3)),ha='center')

plt.savefig("model_comparison.png")

print("Model comparison chart saved as model_comparison.png")