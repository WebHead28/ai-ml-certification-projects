import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

print("Loading Dataset...")

data = pd.read_csv("train.csv")

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset Shape:", data.shape)

print("\nMissing Values:")
print(data.isnull().sum())

plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

X = data.drop("price_range", axis=1)
y = data["price_range"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

print("\nTraining Model...")

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Price Range")
plt.ylabel("Actual Price Range")
plt.show()

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()

from sklearn.preprocessing import label_binarize

y_test_bin = label_binarize(y_test, classes=[0,1,2,3])

y_prob = model.predict_proba(X_test)

fpr = {}
tpr = {}
roc_auc = {}

for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()

for i in range(4):
    plt.plot(fpr[i], tpr[i], label="Class %d (area = %0.2f)" % (i, roc_auc[i]))

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

print("\nEnter mobile features to predict price range:")

battery_power = float(input("Battery Power: "))
ram = float(input("RAM: "))
px_height = float(input("Pixel Height: "))
px_width = float(input("Pixel Width: "))
mobile_wt = float(input("Mobile Weight: "))
talk_time = float(input("Talk Time: "))

sample_dict = {
    "battery_power": battery_power,
    "blue": 1,
    "clock_speed": 2.2,
    "dual_sim": 1,
    "fc": 5,
    "four_g": 1,
    "int_memory": 64,
    "m_dep": 0.6,
    "mobile_wt": mobile_wt,
    "n_cores": 8,
    "pc": 12,
    "px_height": px_height,
    "px_width": px_width,
    "ram": ram,
    "sc_h": 10,
    "sc_w": 5,
    "talk_time": talk_time,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 1
}

sample_df = pd.DataFrame([sample_dict])

sample_scaled = scaler.transform(sample_df)

prediction = model.predict(sample_scaled)

print("\nPredicted Price Range:", prediction[0])

if prediction[0] == 0:
    print("Low Cost Phone")
elif prediction[0] == 1:
    print("Medium Cost Phone")
elif prediction[0] == 2:
    print("High Cost Phone")
else:
    print("Very High Cost Phone")