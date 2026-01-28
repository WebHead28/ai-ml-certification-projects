import pandas as pd

teledata = pd.read_csv("/home/yusuf/Desktop/Ai_Ml_Progs/Telecom_Data.csv")

print(teledata)
print(teledata.head(10))
print(teledata.isnull().sum())
print(teledata.dtypes)

x = teledata.iloc[:, :-1]
y = teledata.iloc[:, -1:]

x = pd.get_dummies(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)

from sklearn.metrics import accuracy_score
a1 = accuracy_score(y_test, y_pred)
print("Accuracy score : {:.2f}%".format(a1*100))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(tn, fp, fn, tp)
