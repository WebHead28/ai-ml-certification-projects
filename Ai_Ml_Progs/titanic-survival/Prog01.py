import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

print(sklearn.__version__)

path = "/home/yusuf/Desktop/Ai_Ml_Progs/titanic.csv"
df = pd.read_csv(path)

print(df.dtypes)

cols_drop = ["Name", "PassengerId", "SibSp", "Parch", "Ticket", "Fare"]
data = df.drop(columns=cols_drop)

data["Age"] = data["Age"].fillna(data["Age"].median())

target = data["Survived"]
features = data.drop(columns=["Survived"])

features = pd.get_dummies(features, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.4, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(classification_report(y_test, y_pred))

score = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", score)
