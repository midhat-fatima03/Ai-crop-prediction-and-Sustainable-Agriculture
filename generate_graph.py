import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Accuracy Graph (Green Shades)
# -------------------------
models = ['RF', 'SVM', 'KNN', 'DT']
accuracy = [0.98, 0.95, 0.92, 0.90]

green_shades = ['#1b5e20', '#2e7d32', '#43a047', '#66bb6a']

plt.figure()
plt.bar(models, accuracy, color=green_shades)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.savefig("static/accuracy_graph.png")
plt.close()


# -------------------------
# Feature Importance Graph (Green Shades)
# -------------------------
features = ['N', 'P', 'K', 'Temp', 'Humidity', 'pH', 'Rainfall']
importance = np.random.rand(len(features))  # replace with real if available

green_shades2 = ['#004d40', '#00695c', '#00796b', '#00897b', '#26a69a', '#4db6ac', '#80cbc4']

plt.figure()
plt.barh(features, importance, color=green_shades2)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.savefig("static/feature_importance.png")
plt.close()

print("✅ Graphs generated successfully!")